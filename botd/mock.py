"""Offline stand-in for AzureLLM, for testing the pipeline end to end (--mock).

Its output goes through the same validators as real output. It deterministically
simulates screen-outs, content-filter blocks, transient failures and leaks.
"""

import hashlib
import re

from .llm import BadOutput, ContentFiltered, TransientError

TESTIMONY = (
    "My name is {FULL_NAME}. I was born in a small town in my home country and worked as a shopkeeper "
    "for eleven years. In March 2019 armed men came to my shop and demanded money every week. I refused "
    "because I could not pay what they asked. They returned on 4 April 2019 and beat me in front of my "
    "customers with sticks and the butts of their rifles. My brother reported the attack to the police "
    "the next morning. The officers wrote nothing down and told him to go home. Two weeks later my brother "
    "was taken from his house at night by men in masks. We never saw him again, and his wife still waits "
    "for news. The armed men telephoned my wife five times and asked where I was sleeping. They said that "
    "{GIVEN_NAME} the shopkeeper would pay with his blood. I closed the shop and moved between the houses "
    "of relatives. I survived because my neighbour hid me in her storeroom for six days. I left my home "
    "country on 12 June 2019 with the help of a smuggler who took all my savings. I stayed in a neighbouring "
    "country for three months, but the men had contacts there too. I arrived in Canada on 20 September 2019 "
    "and asked for protection at the airport. My wife and two children are still hiding with relatives in "
    "the countryside. I fear that the armed men will kill me if I return, because they kill those who refuse "
    "them. I cannot live anywhere else in my country, because the group controls the roads and the police "
    "work with them."
)

ANNOTATION = {
    "hedging": [
        {"original": "In March 2019 armed men", "replacement": "I think it was around March 2019 that armed men"},
        {"original": "on 4 April 2019", "replacement": "on about 4 April 2019"},
        {"original": "for three months", "replacement": "for maybe three months"},
    ],
    "religious": [
        {"original": "I left my home country", "replacement": "I made hijra from my home country"},
        {"original": "I survived because", "replacement": "Alhamdulillah, I survived because"},
    ],
}


def _h(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


class MockLLM:
    def __init__(self, az: dict, labels: dict[str, int]):
        self.az = az
        self.labels = labels  # case_id -> outcome, to simulate reading the decision's conclusion
        self.calls = 0

    def chat_json(self, messages, validate=None):
        self.calls += 1
        system, user = messages[0]["content"], messages[1]["content"]
        if "prepare research material" in system:
            cid = int(re.search(r"case (\d+)", user).group(1))
            if cid % 11 == 0:
                raise ContentFiltered("mock content filter")
            if cid % 7 == 0 and self.calls % 2:
                raise TransientError("mock transient failure")
            if cid % 5 == 0:
                obj = {"usable": False, "reject_reason": "mock: procedural decision"}
            else:
                obj = {
                    "usable": True,
                    "reject_reason": "",
                    "claimant_gender": "female" if cid % 2 else "male",
                    "country_of_origin": "Mockland",
                    "persecution_ground": "criminality/extortion",
                    "claimant_religion": "unknown",
                    # Every 17th case simulates an AsyLex label error.
                    "outcome_in_decision": "granted" if self.labels[str(cid)] ^ (cid % 17 == 0) else "refused",
                    "testimony": TESTIMONY,
                }
        elif "minimal edits" in system:
            obj = ANNOTATION
        else:
            leak = _h(user + str(self.calls)) % 9 == 0
            obj = {"leak": leak, "issues": ["mock leak"] if leak else []}
        problems = validate(obj) if validate else []
        if problems:
            raise BadOutput("; ".join(problems))
        return obj, {"calls": 1, "usage": {}, "repairs": [], "model_version": "mock-1"}
