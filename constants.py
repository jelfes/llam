ACTANTS = [
    "Subject",
    "Object",
    "Sender",
    "Receiver",
    "Helper",
    "Opponent",
]
DEFINITIONS = {
    "Subject": "the character or entity that initiates the action and seeks to achieve a goal",
    "Object": "the target or goal that the subject aims to attain or manipulate",
    "Helper": "the character or entity that assists the subject in achieving its goal",
    "Opponent": "the character or entity that opposes the subject and creates obstacles or conflict",
    "Sender": "the source or initiator of a message or command that sets the action in motion",
    "Receiver": "the entity that receives the message or command from the sender and is expected to respond or act accordingly",
}
MODELS = ["llama_7b", "llama_13b", "llama_70b", "mistral_7b", "llama3_8b", "gemma_7b"]
TEMPLATES = [
    "prompt1_a",
    "prompt1_b",
    "prompt1_c",
    "prompt2_a",
    "prompt2_b",
    "prompt2_c",
]
SEED = 815
SPECIAL_TOKENS = {"NO_MATCH": "[NO_MATCH]", "JSON_ERROR": "[JSON_ERROR]"}
