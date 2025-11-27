# GitHub Copilot Basics (2/2)

- Lesson 1 recap and integrations
- Model Comparison
- Code Review
- GitHub Cloud Products
- Exercise: Legacy refactor

## Lesson 1 recap 🔄

- Role of Copilot as an LLM-based application 🤖
- Measurements and expectations of productivity increase in coding tasks 📈
- Main tools 🛠️
  - Ghost Text completion & Next Edit Suggestion 
  - Inline chat (Ctrl+I) both in the editor and terminal 
  - Ask mode, for dialog 
  - Edit mode, for file changes (even multiple files) with control 
  - Agent mode, for file changes with autonomous capabilities and tool usage 
  - Use of _Chat Participants_ (e.g. `@workspace` or `@vscode`) 🧑
- Use checkpoints to revert to a previous conversation state ⏪
- Use Copilot to generate commit and pull request messages 📝
- Shape Copilot to behave as you want 🧩
  - Repository-wide instructions (`.github/copilot-instructions.md`) 
  - Instructions per glob (`.github/instructions/*.instructions.md`) 
  - Custom prompts (`.github/prompts/*.prompt.md`) 
  - Custom agents (`.github/agents/*.agent.md`) 
- Observe what happens via the Debug View 🐞
- Learned the basics of the MCP protocol 🔌
  - How to add new servers
  - How to manually call specific tools
  - Explored Resources, Prompts, and Custom Tools
- Seen costs and usage limits 💸

### Good to know (lesson 1 addon) 💡

- **Custom prompt** ⚡
  - Refer a tool using `#tool:<tool-name>`
  - Workspace variables - `${workspaceFolder}`, `${workspaceFolderBasename}`
  - File context variables - `${file}`, `${fileBasename}`, `${fileDirname}`, `${fileBasenameNoExtension}`
  - Selection variables - `${selection}`, `${selectedText}`
  - Input variables - `${input:variableName}`, `${input:variableName:placeholder}`
<br/>

- **Built-in prompts (Commands, or Slash Commands)** 🛠️
  - | Command           | Description                                         |
    |-------------------|-----------------------------------------------------|
    | `/clear`          | Start a new chat session. (this is a command)       |
    | `/explain`        | Explain how the code in your active editor works.   |
    | `/fix`            | Propose a fix for problems in the selected code.    |
    | `/fixTestFailure` | Find and fix a failing test.                        |
    | `/help`           | Quick reference and basics of using GitHub Copilot. |
    | `/new`            | Create a new project.                               |
    | `/tests`          | Generate unit tests for the selected code.          |
  - `@vscode /search` is very useful for advanced search or search/replace in the workspace
<br/>

- **Chat variables** 💬
  - | Variable      | Description                                                      |
    |--------------|------------------------------------------------------------------|
    | `#block`     | Includes the current block of code in the prompt.                |
    | `#class`     | Includes the current class in the prompt.                        |
    | `#comment`   | Includes the current comment in the prompt.                      |
    | `#file`      | Includes the current file's content in the prompt.               |
    | `#function`  | Includes the current function or method in the prompt.           |
    | `#line`      | Includes the current line of code in the prompt.                 |
    | `#path`      | Includes the file path in the prompt.                            |
    | `#project`   | Includes the project context in the prompt.                      |
    | `#selection` | Includes the currently selected text in the prompt.              |
    | `#sym`       | Includes the current symbol in the prompt.                       |

- **Custom Agents Handoff** 🤝
  - Handoffs enable you to create guided sequential workflows that transition between agents with suggested next steps. Handoffs are useful for orchestrating multi-step workflows, giving developers control for reviewing and approving each step before moving to the next one.
  - See exercise `02/exercises/00-agents-handoff`
<br/>

- **Terminal integration** 💻
  - Fix, Explain, Attach to Chat
  - `@terminal`, `#terminalLastCommand`, `#terminalSelection` 

#### Awesome Copilot 🚀

Useful resource: <https://github.com/github/awesome-copilot>

## Model Comparison 🤖

The model you choose affects the quality and relevance of responses.

Some aspects that may vary are:

- Latency
- Hallucinations
- Performance on specific tasks
- Capabilities (e.g. multimodality, reasoning)
- Usage limits

New Auto Mode:
- As of today, it supports: 
  - GPT-4.1 (0x)
  - GPT-5 mini (0x)
  - GPT-5 (1x)
  - Claude Haiku 4.5 (0.33x)
  - Claude Sonnet 4.5 (1x)
- 10% discount in case of Premium Request

Model availability changes over time. Refer to GitHub documentation:
- <https://docs.github.com/en/copilot/reference/ai-models/model-comparison>
- <https://docs.github.com/en/copilot/tutorials/compare-ai-models>

## Exercise 01: Code Review 🏋️‍♂️

**Key features**:

- Local analysis of uncommitted changes
- Code quality and style recommendations
- Detection of common security vulnerabilities
- Performance optimization suggestions

**Workflow**:

- Run and explore the application
- Review some sensitive parts (e.g. login) [Command Palette → Chat: Review]
- Create a new `add-announcement-banner` branch
- Add a simple banner feature for teachers to make announcements
    <details>
    <summary>src/static/index.html</summary>

    ```html
    <div class="announcement-banner">
        📢 Activity registration is open until the end of the month. Don't lose your spot!
    </div>
    ```

    </details>
    <details>
    <summary>src/static/styles.css</summary>

    ```css
    .announcement-banner {
        background-color: #4caf50;
        color: white;
        text-align: center;
        padding: 15px;
        font-weight: bold;
    }
    ```

    </details>
- Ask Copilot for feedback [Code Review button in the Changes section] 🧐
  - Tip: Ci sono 3 livelli di review disponibili: `unstaged changes`, `staged changes`, e `uncommitted changes`
- Espandi il pannello dei commenti per trovare la lista dei feedback di Copilot
- Applica e fai commit 💬
- Avvia una nuova pull request e richiedi una Copilot Review direttamente su GitHub 
- Puoi influenzare gli standard di Code Review con le instructions! 📚
  <details>
    <summary>.github/copilot-instructions.md</summary>

    ```markdown
    ## Security

    - Validate input sanitization practices.
    - Search for risks that might expose user data.
    - Prefer loading configuration and content from the database instead of hard coded content. If absolutely necessary, load it from environment variables or a non-committed config file.

    ## Code Quality

    - Use consistent naming conventions.
    - Try to reduce code duplication.
    - Prefer maintainability and readability over optimization.
    - If a method is used a lot, try to optimize it for performance.
    - Prefer explicit error handling over silent failures.
    ```

    </details>
    <details>
    <summary>.github/instructions/frontend.instructions.md</summary>

    ```markdown
    ---
    applyTo: "*.html,*.css,*.js"
    ---

    ## Frontend Guidelines

    - Use accessibility attributes (alt text, aria labels) and color schemes.
    - Use responsive design for compatibility with mobile devices.
    - Validate HTML structure and semantic elements
    ```

    </details>
    <details>
    <summary>.github/instructions/backend.instructions.md</summary>

    ```markdown
    ---
    applyTo: "backend/**/*,*.py"
    ---

    ## Backend Guidelines

    - All API endpoints must be defined in the `routers` folder.
    - Load example database content from the `database.py` file.
    - Error handling is only logged on the server. Do not propagate to the frontend.
    - Ensure all APIs are explained in the documentation.
    - Verify changes in the backend are reflected in the frontend (`src/static/**`). If possible breaking changes are found, mention them to the developer.
    ```

    </details>
- Prova a richiedere una nuova Code Review 📝

## GitHub Cloud Products ☁️

A brief overview of some recently released tools: 

- GitHub Coding Agent
  - Analyzing a real life example
- GitHub Spaces
  - Chatting with the BOM repo
- GitHub Spark
  - Fast demo of a vibe-coded tool

## Exercise 02: Refactor a legacy app 🏋️‍♂️

Welcome in the early 1990s!

You have to rewrite with modern technlogies an old COBOL-based accounting system.
Copilot can help in this process by:

1. Helping decipher the decades-old COBOL code that lacks documentation.
2. Assisting with test creation to ensure business logic remains intact.
3. Translating COBOL structures to modern Node.js equivalents.
4. Testing the new code to ensure it meets the original system's requirements.

**Workflow**:

- Take a few minutes to explore the COBOL files in the repository
  <details>
    <summary>compile and run the application</summary>

    ```bash
    cobc -x src/cobol/main.cob src/cobol/operations.cob src/cobol/data.cob -o accountsystem
    ./accountsystem
    ```

    </details>
- Add the COBOL files to the context and ask Copilot to explain the purpose of each file in the context 
  <details>
    <summary>example prompt</summary>

    ```markdown
    Create a README.md file in a new /docs directory

    Document the purpose of each COBOL file, key functions, and any specific business rules related to student accounts.
    ```

    </details>
- > Notice how we are breaking down the task into smaller steps.
  > You will find that Copilot is more effective when you provide it with specific smaller tasks rather than trying to do everything at once, e.g `Hey Copilot, refactor this COBOL codebase to Node.js`. This is especially true when working on large codebase modernizations and context window limitations come into play.
- Ask Copilot to create a Mermaid data flow diagram that illustrates how data moves through the accounting system
  <details>
    <summary>example prompt</summary>

    ```markdown
    Create a sequence diagram of the app showing the data flow of the app.

    Please create this in mermaid format so that I can render this at the end of the the docs/README.md markdown file.
    ```

    </details>
- Generate a comprehensive test plan that covers all critical functionalities and edge cases
  <details>
    <summary>example prompt</summary>

    ```markdown
    The current Cobol app has no tests.
    Create a test plan of the current business logic and implementation that I can use to validate with business stakeholders.
    Store it in a file called docs/TESTPLAN.md.
    Later I would like to use this test plan to create unit and integration tests in a node.js app. I am in the middle of transforming the current Cobol app to a node.js app.
    The test plan should include the following headings:
    1. Test Case ID
    2. Test Case Description
    3. Pre-conditions
    4. Test Steps
    5. Expected Result
    6. Actual Result
    7. Status (Pass/Fail)
    8. Comments

    Please create the test plan in a markdown table format. The test plan should cover all the business logic in the current Cobol app.           
    ```

    </details>
- Leverage Copilot to transform COBOL to Node.js
  <details>
    <summary>example prompt</summary>

    ```markdown
    #codebase convert the three separate COBOL legacy files into a single Node.js src/accounting/index.js accounting application.

    Leverage the data flow diagram of the existing COBOL application available in the repository to preserve:
    - the original business logic
    - data integrity
    - menu options of the original application.

    Change directory to src/accounting and install all prerequisites to run the Node.js application

    Create a .vscode/launch.json file to run the Node.js application         
    ```

    </details>
- Create Unit Tests based on our Test Plan
  <details>
    <summary>example prompt</summary>

    ```markdown
    #codebase change directory to src/accounting and install all prerequisites for the test framework.

    - Write unit tests for the Node.js application that mirror the scenarios in the testplan.
    - Place the tests in a dedicated test file.
    - Make sure each test checks the expected behavior described in the COBOL test plan.        
    ```

    </details>

## Keep yourself updated!

- https://github.blog/changelog/?label=copilot