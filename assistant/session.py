import time
import re
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4
from .model import ScriptedConversation
from .policy import Decision, classify
from .state import Machine, State
from .telephony import SimulatedTransport
from .adapters import ScriptedAdapter, ModelAdapter, ModelUnavailable
from .audit import safe_diagnostic
from .diagnostics import exception_metadata, safe_metadata
from .proposals import Proposal, ProposalRejected, CATALOG, validate_proposal


@dataclass(frozen=True)
class PendingQuestion:
    id: str
    text: str
    blocked: bool
    fact_ids: tuple[str, ...] | None = None


class Session:
    def __init__(self, objective, *, adapter: ModelAdapter | None = None):
        objective.validate()
        self._objective = objective
        self.machine = Machine()
        self.model = ScriptedConversation()
        self.adapter = adapter if adapter is not None else ScriptedAdapter()
        self.proposals = []
        self._clarifications = 0
        self.transport = SimulatedTransport()
        self.transcript = []
        self.index = 0
        self.asked = []
        self.answers = []
        self._pending = None
        self._messages = {}
        self._question_id = None
        self._spoken = []
        self.started = None
        self.finished = None
        self.end_reason = None
        self.details = {"organization_person_reached": [], "reference_numbers": [],
                        "required_next_actions": [], "deadlines_mentioned": []}

    @property
    def objective(self):
        return self._objective

    @property
    def pending(self):
        return self._pending

    @property
    def question_id(self):
        return self._question_id

    @property
    def state(self):
        return self.machine.state

    def record(self, source, text):
        self.transcript.append({"source": source, "text": safe_diagnostic(text) if isinstance(text, str) else text})

    def speak(self, *args, **kwargs):
        raise ValueError("Arbitrary speech is disabled; use the authorized session operations.")

    def _emit(self, action, fact=None, *, pending_id=None, proposal=None):
        # Only trusted application code calls this method. Remote strings never select actions.
        if self.state not in {State.CONNECTED, State.LISTENING, State.ASK_USER}:
            raise ValueError("Speech is unavailable in this state.")
        if not self.transport.connected:
            self.disconnect()
            raise ValueError("Transport disconnected.")
        if action == "fact" and (self.state != State.ASK_USER or self.pending is None
                                 or self.pending.blocked or pending_id != self.pending.id):
            raise ValueError("Fact speech requires authorization for this pending question.")
        if action == "introduction":
            text = self.model.introduction(self.objective)
        elif action == "question":
            text = self.model.question(self.objective.questions[self.index])
        elif action == "fact" and type(fact) is int and 0 <= fact < len(self.objective.facts):
            text = "The person I'm assisting provided this fact: " + self.objective.facts[fact]
        elif action in {"decline", "hold", "goodbye"}:
            text = getattr(self.model, action)()
        elif (action == "proposal" and type(proposal) is Proposal
              and proposal.intent in {"SPEAK", "CLARIFY"}
              and not proposal.requested_fact_ids
              and proposal.proposed_text in CATALOG[proposal.intent]
              and self.state == State.LISTENING and self.pending is None):
            text = proposal.proposed_text
        else:
            raise ValueError("Unknown speech action.")
        self.machine.move(State.AGENT_SPEAKING)
        self._spoken.append({"action": action, "text": text})
        self.record("APPROVED MODEL SPEECH" if action == "proposal"
                    else "MODEL-GENERATED CONVERSATION (scripted)", text)
        self.machine.move(State.ASK_USER if action == "hold" else State.LISTENING)

    def start(self, confirmed=False):
        if confirmed is not True:
            raise ValueError("Explicit objective confirmation is required.")
        self.machine.move(State.DIALING)
        self.started = time.monotonic()
        self.transport.connect()
        self.machine.move(State.CONNECTED)
        self._emit("introduction")
        self.ask_next()

    def ask_next(self):
        if self.state != State.LISTENING or self.pending is not None:
            raise ValueError("Cannot advance questions in this state.")
        if self.index == len(self.objective.questions):
            self.abort(reason="completed")
            return
        self._question_id = uuid4().hex
        self._clarifications = 0
        self._emit("question")
        self.asked.append(self.objective.questions[self.index])

    def remote(self, kind, text, *, message_id, question_id):
        if (not isinstance(message_id, str) or not message_id.strip()
                or not isinstance(question_id, str) or not question_id.strip()
                or not isinstance(kind, str) or not kind.strip()
                or not isinstance(text, str) or not text.strip() or len(text) > 8192):
            raise ValueError("Remote events require nonempty text and identifiers.")
        payload = (kind, text, question_id)
        if message_id in self._messages:
            if self._messages[message_id] != payload:
                raise ValueError("Message ID reused with different content.")
            return False  # Exact replay: no speech, state change, or transcript duplication.
        if self.state != State.LISTENING or self.pending is not None:
            raise ValueError("Remote input is disabled until the user responds.")
        if not self.transport.connected:
            self.disconnect()
            raise ValueError("Transport disconnected.")
        if question_id != self.question_id:
            raise ValueError("Stale question identifier.")
        decision = classify(kind, text)
        self._messages[message_id] = payload
        if decision != Decision.ALLOW:
            blocked = decision == Decision.BLOCK
            display = "[Prohibited or sensitive request: decline or abort only]" if blocked else text
            self.record("REMOTE (untrusted request)", display)
            self._pending = PendingQuestion(uuid4().hex, display, blocked)
            if blocked:
                self._emit("decline")
            self._emit("hold")
        else:
            self.record("REMOTE (untrusted answer)", text)
            self._model_turn(safe_diagnostic(text), question_id, message_id)
        return True

    def _pause_model(self, text, *, blocked, fact_ids=()):
        self._pending = PendingQuestion(uuid4().hex, text, blocked, fact_ids)
        self._emit("hold")

    def _model_turn(self, text, question_id, message_id):
        fact_ids = tuple(f"F{i + 1}" for i in range(len(self.objective.facts)))
        context = {
            "objective": {"purpose": self.objective.purpose},
            "questions": list(self.objective.questions),
            "current_question": self.objective.questions[self.index],
            "approved_facts": dict(zip(fact_ids, self.objective.facts)),
            "remote_input_untrusted": text,
        }
        raw = None
        error = None
        diagnostic = {}
        try:
            raw = self.adapter.propose(deepcopy(context))
        except Exception as exc:
            error = "adapter_error"  # Never log provider exceptions or credentials.
            diagnostic = {"parser_failure_reason": "not_attempted", **exception_metadata(exc)}
            if type(exc) is ModelUnavailable:
                diagnostic.update(safe_metadata(exc.diagnostic))
        else:
            diagnostic = safe_metadata(getattr(self.adapter, "last_diagnostic", {}))
            try:
                proposal = validate_proposal(raw, fact_ids)
            except ProposalRejected as exc:
                error = str(exc)  # Only this local validator can supply diagnostic codes.
                diagnostic["parser_failure_reason"] = error
        audit = {"message_id": message_id, "question_id": question_id,
                 "raw_proposal_untrusted": ("[No proposal returned: adapter failed]" if error == "adapter_error"
                                            else "[Model content REDACTED from audit]"),
                 "decision": "rejected" if error else "validated", "reason": error,
                 "diagnostic": safe_metadata(diagnostic)}
        self.proposals.append(audit)
        # A result has no authority if local state changed while the adapter was running.
        if (self.state != State.LISTENING or self.pending is not None
                or self.question_id != question_id):
            audit.update(decision="rejected", reason="stale_model_result")
            return
        if not self.transport.connected:
            audit.update(decision="rejected", reason="transport_disconnected")
            self.disconnect()
            return
        if error:
            self._pause_model("Model proposal rejected. Decline to resume or ABORT.", blocked=True)
            return
        if proposal.intent == "ASK_USER":
            audit.update(decision="hold_only", reason="local_user_response_required")
            self._pause_model("Review remote input (untrusted): " + text,
                              blocked=False, fact_ids=proposal.requested_fact_ids)
        elif proposal.intent == "END":
            # A model's claim that the conversation is complete is insufficient.
            if text.strip().lower().rstrip(".!") not in {"goodbye", "bye"}:
                audit.update(decision="rejected", reason="end_not_supported_by_remote_farewell")
                self._pause_model("Model proposed ending without a farewell. Decline or ABORT.", blocked=True)
            else:
                audit.update(decision="approved", reason="remote_farewell")
                self.abort(reason="remote_farewell")
        elif proposal.intent == "CLARIFY":
            if self._clarifications >= 3:
                audit.update(decision="rejected", reason="clarification_limit")
                self._pause_model("Clarification limit reached. Decline or ABORT.", blocked=True)
            else:
                self._emit("proposal", proposal=proposal)
                self._clarifications += 1
                self._question_id = uuid4().hex
                audit.update(decision="approved", reason="catalog_clarification")
        else:
            self._emit("proposal", proposal=proposal)
            audit.update(decision="approved", reason="catalog_acknowledgment")
            self.answers.append({"question": self.objective.questions[self.index], "answer": text})
            self.index += 1
            self.ask_next()

    def respond(self, response, *, pending_id):
        if self.state != State.ASK_USER or self.pending is None:
            raise ValueError("No user response is pending.")
        if not isinstance(pending_id, str) or pending_id != self.pending.id:
            raise ValueError("Response does not match the pending question.")
        if not isinstance(response, str):
            raise ValueError("Response must be a command string.")
        fact = None
        if response == "decline":
            action = "decline"
        elif self.pending.blocked:
            raise ValueError("This request permits decline or ABORT only.")
        elif re.fullmatch(r"fact [1-9][0-9]{0,5}", response):
            fact = int(response[5:]) - 1
            if fact >= len(self.objective.facts):
                raise ValueError("Choose a valid approved fact number.")
            if self.pending.fact_ids is not None and f"F{fact + 1}" not in self.pending.fact_ids:
                raise ValueError("That fact was not requested for this pending question.")
            action = "fact"
        else:
            raise ValueError("Use 'fact N', 'decline', or ABORT. New free-form disclosures are disabled.")
        self._emit(action, fact, pending_id=pending_id)
        self.record("USER AUTHORIZATION", {"pending_id": pending_id, "response": response})
        self._pending = None
        self._question_id = uuid4().hex  # Invalidate queued remote answers from before the hold.
        self._emit("question")

    def _finish(self, reason):
        self._pending = None
        self._question_id = None
        self.transport.disconnect()
        self.finished = time.monotonic()
        self.end_reason = reason
        self.machine.move(State.ENDED)

    def abort(self, reason="aborted"):
        if self.state == State.ENDED:
            return
        if self.state in {State.CONNECTED, State.LISTENING, State.ASK_USER} and self.transport.connected:
            self._emit("goodbye")
        if self.state != State.ENDING:
            self.machine.move(State.ENDING)
        self._finish(reason)

    def disconnect(self):
        if self.state == State.ENDED:
            return
        # No goodbye can be delivered after an unexpected disconnect.
        if self.state != State.ENDING:
            self.machine.move(State.ENDING)
        self.record("LOCAL EVENT", "Unexpected simulated disconnect")
        self._finish("disconnected")

    def summary(self):
        # Scripted actions cannot authorize commitments. User wording is not semantically certified.
        supplied_speech = [row["text"] for row in self._spoken
                           if row["action"] in {"introduction", "question", "fact"}]
        return deepcopy({"mode": "offline simulation", "call_duration_seconds":
                round((self.finished if self.finished is not None else time.monotonic()) - self.started, 2)
                if self.started is not None else 0,
                **self.details, "questions_asked": self.asked, "answers_received": self.answers,
                "unresolved_questions": list(self.objective.questions[self.index:]),
                "commitments_made_by_assistant": "None authorized by application",
                "commitment_content_verified": False,
                "user_supplied_speech_for_review": supplied_speech,
                "end_reason": self.end_reason,
                "summary_provenance": "Remote answers are untrusted, not evidence of assistant commitments. "
                "Fixed actions never authorize commitments; user-supplied wording requires human review. "
                "Metadata is user-entered, not inferred."})
