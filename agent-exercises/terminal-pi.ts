import { Agent } from "@earendil-works/pi-agent-core";
import readline from "readline";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const agent = new Agent({
  // Uses pi's minimal system prompt & tools
  systemPrompt: "Execute terminal commands based on natural language input.",
});

function promptUser() {
  rl.question("english-sh> ", async (input) => {
    if (input.trim() === "exit") process.exit(0);

    // Stream response and execute bash actions
    await agent.run(input);
    promptUser();
  });
}

promptUser();