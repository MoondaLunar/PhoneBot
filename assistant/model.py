"""Fixed offline utterances; intentionally no OpenAI integration."""


class ScriptedConversation:
    def introduction(self, objective):
        return (f"Hello, I am an AI voice assistant calling on behalf of {objective.name}. "
                f"The purpose is: {objective.purpose}. I can only gather information.")

    def question(self, text):
        return text

    def hold(self):
        return "I need to confirm that with the person I'm assisting. One moment."

    def decline(self):
        return "I cannot authorize actions, make commitments, or disclose sensitive information."

    def goodbye(self):
        return "Thank you for your time. I am ending the call now. Goodbye."
