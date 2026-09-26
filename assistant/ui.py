import argparse
import json
import os
from pathlib import Path
from uuid import uuid4
from .policy import Objective, SYSTEM_RULES, UNSAFE
from .session import Session
from .state import State
from .audit import safe_diagnostic, redact_credentials


def main():
    parser = argparse.ArgumentParser(description="Offline phone assistant; cannot place real calls")
    parser.add_argument("objective", type=Path, help="Local objective JSON")
    parser.add_argument("--save", type=Path, help="Optional new local JSON transcript/summary file")
    parser.add_argument("--adapter", choices=("scripted", "openai", "ollama"), default=None)
    parser.add_argument("--allow-api", action="store_true", help="Explicitly permit billable text requests to OpenAI")
    parser.add_argument("--model", help="Model identifier for the selected model adapter")
    parser.add_argument("--ollama-host", help="Ollama HTTP endpoint (default: OLLAMA_HOST or localhost)")
    args = parser.parse_args()
    backend = args.adapter or os.environ.get("PHONEBOT_MODEL_BACKEND", "scripted")
    if backend not in {"scripted", "openai", "ollama"}:
        parser.error("PHONEBOT_MODEL_BACKEND must be scripted, openai, or ollama.")
    if (backend == "openai") != args.allow_api:
        parser.error("OpenAI mode requires both --adapter openai and --allow-api.")
    try:
        data = json.loads(args.objective.read_text(encoding="utf-8"))
        objective = Objective(data["destination"], data["name"], data["purpose"],
                              data["questions"], data.get("facts", []))
        adapter = None
        if backend == "openai":
            from .openai_model import OpenAIRealtimeAdapter
            adapter = OpenAIRealtimeAdapter(allow_api=True, **({"model": args.model} if args.model else {}))
        elif backend == "ollama":
            from .ollama_model import OllamaModelAdapter
            options = {"model": args.model} if args.model else {}
            if args.ollama_host:
                options["host"] = args.ollama_host
            adapter = OllamaModelAdapter(**options)
        session = Session(objective, adapter=adapter)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print("SIMULATED CALL - no audio or real phone call")
    if backend == "openai":
        print("OPENAI MODE: after CONFIRM, eligible turns send purpose, questions, approved facts,")
        print("and current remote text to OpenAI. Text requests may incur API charges.")
        print("During a request, Ctrl+C aborts; otherwise requests time out after 20 seconds.")
    elif backend == "ollama":
        print("OLLAMA MODE: after CONFIRM, eligible turns send purpose, questions, approved facts,")
        print("and current remote text to the configured Ollama endpoint.")
        print("No fallback to OpenAI is performed; requests time out after 20 seconds.")
    else:
        print("SCRIPTED MODE - fully offline")
    print("\nSYSTEM RULES\n" + SYSTEM_RULES)
    print("\nUSER-SUPPLIED FACTS / COMPLETE OBJECTIVE\n" + redact_credentials(json.dumps(data, indent=2)))
    print("\nQUESTIONS\n" + redact_credentials("\n".join(objective.questions)))
    print("\nConfirm destination, name, purpose, questions, and every permitted fact.")
    try:
        if input("Type CONFIRM to begin: ").strip() != "CONFIRM":
            print("Cancelled. No session started.")
            return
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return
    session.start(confirmed=True)
    print("Remote input: answer TEXT or request TEXT (unprefixed input is a request).")
    print("Local commands: ABORT, /disconnect, /note organization|reference|action|deadline TEXT.")
    print("Notes are local user annotations, not transmitted. Do not enter secrets.")
    cursor = 0
    proposal_cursor = 0

    def display():
        nonlocal cursor, proposal_cursor
        for row in session.proposals[proposal_cursor:]:
            print("MODEL PROPOSAL AUDIT (untrusted, NOT speech):", json.dumps(row, ensure_ascii=True))
        proposal_cursor = len(session.proposals)
        for row in session.transcript[cursor:]:
            # Escape terminal controls from untrusted input.
            print(f"{row['source']}: {json.dumps(row['text'], ensure_ascii=True)}")
        cursor = len(session.transcript)

    while session.state != State.ENDED:
        display()
        try:
            if session.state == State.ASK_USER:
                print("ASK_USER:", json.dumps(session.pending.text))
                for number, fact in enumerate(objective.facts, 1):
                    if not session.pending.blocked and (session.pending.fact_ids is None
                                                       or f"F{number}" in session.pending.fact_ids):
                        print(f"  fact {number}: {json.dumps(safe_diagnostic(fact))}")
                print("Only a typed fact selection for this question authorizes disclosure; decline or ABORT are available.")
            line = input(f"[{session.state.value}] > ").strip()
            if line.upper() == "ABORT":
                session.abort()
            elif line == "/disconnect":
                session.disconnect()
            elif line.startswith("/note "):
                category, _, value = line[6:].partition(" ")
                keys = {"organization": "organization_person_reached", "reference": "reference_numbers",
                        "action": "required_next_actions", "deadline": "deadlines_mentioned"}
                if category not in keys or not value or UNSAFE.search(value):
                    raise ValueError("Invalid or potentially sensitive note.")
                session.details[keys[category]].append(value)
            elif session.state == State.ASK_USER:
                session.respond(line, pending_id=session.pending.id)
            else:
                kind, _, value = line.partition(" ")
                session.remote(kind, value if kind in {"answer", "request"} else line,
                               message_id=uuid4().hex, question_id=session.question_id)
        except (EOFError, KeyboardInterrupt):
            session.abort()
        except ValueError as error:
            print("Input rejected:", error)
    display()
    result = {"summary": session.summary(), "transcript": session.transcript,
              "model_proposals": session.proposals}
    print("\nSUMMARY\n" + redact_credentials(json.dumps(result["summary"], indent=2)))
    if args.save:
        try:
            with args.save.open("x", encoding="utf-8") as stream:
                stream.write(redact_credentials(json.dumps(result, indent=2)))
            print("Saved locally:", args.save.resolve())
        except OSError as error:
            print("Could not save:", error)


if __name__ == "__main__":
    main()
