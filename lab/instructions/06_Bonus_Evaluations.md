# Bonus: Manually Evaluate Your Agent Responses

> [!NOTE]
> This is a bonus section you can complete if you still have time during the allotted lab slot. Otherwise, you are more than welcome to go through it at your own pace once back home.

In this section, you will learn how to manually evaluate a dataset of your agent's responses. Manual evaluations are when humans directly judge the quality of an LLM’s output. In practice, this means a person reads the generated response and decides—often against a rubric or simple scale—whether it is correct, relevant, clear, or “good” versus “bad.” With Agent Builder, you can complete manual evaluations to assess your agent’s performance.

## Step 1: Add a Variable to the Agent Instructions

To use the Evaluation features of Agent Builder, the agent's **Instructions** must contain a variable. The variable itself is a value that can change the context of the agent instructions or user prompt, but is still relevant to the general purpose of the agent. Variables are surrounded by 2 sets of curly braces (ex: `{{variable}}`).

Since the Cora agent's general purpose is to support store operations and head office reporting, it makes most sense to use a variable that changes the operational context. In this example, we’ll use **store** as a variable. What we can do is modify the **Instructions** to the following:

```
You are Cora, an internal assistant for Zava. You help store managers and head office staff analyze sales and manage inventory, tailored to the needs of the {{store}} location.​

Your role is to:​

- Ask clarifying questions and be brief in your responses.​

- Use Zava’s tools (sales + inventory) to answer questions with facts when possible.​

- Summarize sales performance, answer inventory questions, and recommend next actions for the {{store}} location.​
​
Your personality is:​

- Professional, precise, and helpful​

- Curious and practical—never assume, always clarify​
```

> [!NOTE]
> Make sure the Model is still set to **gpt-4o (via Microsoft Foundry)**.

All variables are stored in the **Variables** section in Agent Builder. ignore the error message you see in the screenshot below, as we are going to pass values for the variable through the Evaluation tab.

![Agent variables.](../../img/agent-variables.png)

So how does this work? Suppose we want to use `Seattle` as the `{{store}}`. Assuming you've defined `Seattle` as the value for `{{store}}`, when the user prompt is run, the **Instructions** are dynamically modified to reflect the value `Seattle` for the `{{store}}` variable. Therefore, the agent instructions would read:

"You are Cora, an internal assistant for Zava. You help store managers and head office staff analyze sales and manage inventory, tailored to the needs of the Seattle location.​"

Let's see this in action by running a few lines of evaluation data!

## Step 2: Add Data

In Agent Builder, switch to the **Evaluation** tab. Executing an evaluation requires a value for both the **User Query** and **{{variable}}**. The **User Query** is the prompt that the user submits to the agent (ex: What were my top categories last month?). The **{{variable}}** is the value for your variable (ex: `{{store}}`).

> [!NOTE]
> The {{store}} variable will only show in the table header after clicking **+ Add an Empty Row**.
>

![Evaluation table.](../../img/evaluation-table.png)

You have a couple of options from here with respect to how you'd like to add data for your evaluation.

> [!TIP]
> To expand the **Evaluation** section, click the **Expand to Full Screen** icon next to the Trash Can icon.

**Manually Add Data**

You can manually add your own data in the **Evaluation** tab, by creating an empty row and adding input for the **User Query** and **{{store}}** cells. Provided below are some examples of **User Query** and **{{store}}** pairs:

|   User Query        | {{store}}
--------------|-------------
What were the top 3 categories by revenue last month? | Seattle
Which products are at risk of stockout this week? | Redmond
Summarize online vs physical sales performance last month. | Head Office
Do we have enough circuit breakers for this weekend’s promotion? | Bellevue

> [!TIP]
> Use the **Add an Empty Row button** to create each row of the table and then double-click on a cell to edit its content.

**Generate Data**

If you need help with creating data, the **Generate Data** feature can generate up to 10 rows of synthetic data. Synthetic data is artificially created data that mimics real-world information, but isn’t collected from actual people or events. The feature itself leverages a LLM that takes **Generation Logic** as input to create **User Query** and **{{store}}** pairs. The **Generate Data** feature generates its own set of instructions (or Generation Logic) based on the agent's **Instructions**. However, you can modify the **Generation Logic** to your liking.

![Generate data.](../../img/generate-data.png)

After entering the number of **Rows of Data to Generate**, modify the **Generation Logic** and select **Generate** to generate a dataset. The generated dataset appears in the evaluation table.

**Import a Dataset**

If you've created your own bulk dataset of **User Query** and **{{store}}** pairs, you could import the dataset to Agent Builder for evaluation. Agent Builder supports `.csv` files that are formatted in the following manner:

|   User Query        | {{store}}
--------------|-------------
What were the top 3 categories by revenue last month? | Seattle
Which products are at risk of stockout this week? | Redmond
Summarize online vs physical sales performance last month. | Head Office
Do we have enough circuit breakers for this weekend’s promotion? | Bellevue

Whereas both **User Query** and **{{store}}** are headers. The **Import** icon (i.e. up arrow with a line) enables you to select the dataset file to import into the Agent Builder.

![Import dataset.](../../img/import-dataset.png)

Consider experimenting with each option! The remaining instructions for this lab will continue to follow the first option: **Manually Add Data**

## Step 3: Assess Your Agent Output

With your dataset prepared, you can run rows one by one or select multiple rows to run together. To select all rows, check the box in the header row. To run the selected rows, select the **Run Response** icon (i.e. play button).

![Run button.](../../img/run-eval.png)

The model will generate a response for each **User Query** and **{{store}}** pair. Once the response is generated, review the output and select either the **thumbs up** or **thumbs down** icon in the **Manual** column.

![Manual evaluation.](../../img/manual-evaluation.png)

How do you decide whether the response deserves a **thumbs up** or **thumbs down**? When deciding whether to give a thumbs up or thumbs down, think about whether the output met your expectations. A **thumbs up** means the response was accurate, relevant, clear, and genuinely helpful—it gave you the information or result you were looking for. A **thumbs down** means the response fell short in some way, such as being incorrect, incomplete, confusing, off-topic, or not useful for your task.

In short, ask yourself: **Did the output do what I needed it to? If yes, choose thumbs up; if not, choose thumbs down.**

## Key Takeaways

- Adding variables like {{store}} to agent instructions allows for systematic testing across different operational contexts while maintaining the agent's core purpose and functionality.
- Agent Builder supports manual data entry, synthetic data generation, and CSV imports, providing flexibility for creating evaluation datasets that match specific testing needs.
- Human judgment through thumbs up/down ratings helps assess whether agent responses meet expectations for accuracy, relevance, and usefulness beyond automated metrics.