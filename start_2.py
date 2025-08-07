import ollama


response = ollama.list()

# print(response)

# == Chat example ==
res = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "why is the sky blue?"},
    ],
)

# print(res["message"]["content"])

# == Chat example streaming ==
res = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "why is the ocean so salty?",
        },
    ],
    stream=True,
)

for chunk in res:
    print(chunk["message"]["content"], end="", flush=True)
