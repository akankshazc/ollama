import ollama


response = ollama.list()

# print(response)

# == Chat example ==
res = ollama.chat(
    model="gemma3:4b",
    messages=[
        {"role": "user", "content": "why is the sky blue?"},
    ],
)

# print(res["message"]["content"])

# == Chat example streaming ==
res = ollama.chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "user",
            "content": "why is the ocean so salty?",
        },
    ],
    stream=True,
)

# for chunk in res:
#     print(chunk["message"]["content"], end="", flush=True)


# == Generate example ==
res = ollama.generate(
    model="gemma3:4b",
    prompt="why does the rainbow so colorful? be concise.",
)

# show
print(ollama.show("gemma3:4b"))


# == Create a new model with modelfile ==
file = """
FROM gemma3:4b
SYSTEM You are Tessa, a very smart assistant who answers questions succintly and accurately.
PARAMETER temperature 0.3
"""

# ollama.create doesn't accept modelfile anymore, so we need to use CLI only
# ollama.create(model="tessa", modelfile=file)

# res = ollama.generate(model="tessa", prompt="what is the meaning of life?")

# print(res["response"])
