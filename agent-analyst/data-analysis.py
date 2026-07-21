import base64

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

print("Running code interpreter sandbox...")
response = client.responses.create(
    model="gpt-4o-mini",
    tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
    include=["code_interpreter_call.outputs"],
    input=(
        "Using the code interpreter tool, execute Python code now to: load the iris dataset "
        "from sklearn.datasets, clean any missing data, compute the correlation between "
        "features, and save a Seaborn heatmap plot as iris_heatmap.png."
    ),
)

saved = False

for item in response.output:
    if item.type == "message":
        for content in item.content:
            print(f"\nAssistant Response:\n{content.text}")
            # Fallback path: the sandbox sometimes reports the saved file only
            # as a citation on the message, not as inline output below.
            for annotation in getattr(content, "annotations", []):
                if getattr(annotation, "type", None) == "container_file_citation" and annotation.filename == "iris_heatmap.png":
                    image_bytes = client.containers.files.content.retrieve(
                        annotation.file_id, container_id=annotation.container_id
                    ).read()
                    with open("iris_heatmap.png", "wb") as f:
                        f.write(image_bytes)
                    saved = True
    elif item.type == "code_interpreter_call":
        # Fast path: the sandbox sometimes returns the image inline as base64.
        for output in item.outputs or []:
            if output.type == "image":
                image_bytes = base64.b64decode(output.url.split(",", 1)[1])
                with open("iris_heatmap.png", "wb") as f:
                    f.write(image_bytes)
                saved = True

print("Saved plot to iris_heatmap.png" if saved else "No plot file was produced.")
