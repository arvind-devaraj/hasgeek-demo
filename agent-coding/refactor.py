from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool

load_dotenv()

BEFORE_PATH = "before.py"
AFTER_PATH = "after.py"


@tool
def write_file(path: str, content: str) -> str:
    """Writes content to a file.

    Args:
        path: Path of the file to write.
        content: The text content to write into the file.
    """
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {path}"


# smolagents' local executor deliberately blocks the raw `open()` builtin as
# a safety boundary, so file I/O has to go through an explicit, whitelisted
# tool like write_file above instead of arbitrary code touching the disk.
agent = CodeAgent(
    model=OpenAIServerModel("gpt-4o-mini"),
    tools=[write_file],
)

if __name__ == "__main__":
    agent.run(
        f"Here is the content of {BEFORE_PATH}:\n\n{open(BEFORE_PATH).read()}\n\n"
        f"Refactor it to remove the hardcoded URL and secret, replacing them with "
        f"values read from a config dict at the top of the file. Use the write_file "
        f"tool to save the refactored version to '{AFTER_PATH}'."
    )

    print("\n" + "=" * 50)
    print(f"BEFORE ({BEFORE_PATH}):\n" + "-" * 50)
    print(open(BEFORE_PATH).read())
    print(f"AFTER ({AFTER_PATH}):\n" + "-" * 50)
    print(open(AFTER_PATH).read())
    print("=" * 50)
