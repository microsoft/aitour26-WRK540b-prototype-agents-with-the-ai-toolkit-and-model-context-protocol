# Introduction

> [!NOTE]
>This is a **75-minute** workshop that will give you hands-on experience with the AI Toolkit (AITK) and Microsoft Foundry in Visual Studio Code to prototype multimodal agents for your business scenario.

## Learning Objectives

By the end of this workshop, you should be able to:
- Explore and compare models in the AITK Model catalog, to select the best fit for your use-case.
- Augment models with prompts and data to get more accurate and grounded responses in the AITK Playground.
- Prototype an agent by combining models and instructions with tools via MCP (Model Context Protocol) using the AITK Agent Builder.

## Resources

> [!TIP]
> You can find login and subscription information over on the Resources tab.

## Lab Outline

The lab is organized into 4 sections, taking you through the process of prototyping a multimodal agent with Microsoft Foundry and the AI Toolkit.

1. **Part 1 - Model Selection** Model selection is an essential step in building AI solutions. In this section, you will explore the AI Toolkit Model Catalog to compare and select models that best fit your business scenario.
2. **Part 2 - Model Augmentation** Once you have selected a model, you will learn how to augment it using prompt engineering and context data to improve its performance and relevance to your specific use case.
3. **Part 3 - Agent Prototyping** In this section, you will use the AITK Agent Builder to prototype an agent. You will combine your selected and augmented models with instructions and tools via MCP (Model Context Protocol).
4. **Part 4 - From Prototype to Code** Finally, you will learn how to export your agent prototype into code that can be integrated into your applications.

## Business Scenario

In this workshop, you'll be building an AI agent for **Zava**, a DIY retailer with **20 stores across the United States** and an online channel. The scenario focuses on **store operations** and **head office sales analysis**, including the ability to check inventory and move stock between stores.

### The Challenge

Zava store managers and head office teams need fast answers to questions like:
- “What were the top-selling categories last month?”
- “Which stores are understocked on a key item?”
- “Can we move inventory from one store to another to meet demand?”

They also need to handle occasional multimodal tasks, like identifying an item from a photo (for example, a circuit breaker), then checking availability and taking action.

### The Solution: Cora, Zava's Store Ops & Sales Analysis Agent

You'll be developing **Cora**, an internal-facing assistant that can:

1. **Understand multimodal inputs**: Process text and images from employees (e.g., photos of a product or SKU label)

2. **Search the product catalog**: Find relevant products based on natural language queries (and, when needed, an image-based description)

3. **Support operational decisions**: Answer questions about sales performance and provide actionable summaries

4. **Check availability**: Access real-time inventory data for both online and physical stores

5. **Move inventory safely**: Initiate stock transfers between stores (with explicit confirmation)

### Why This Matters

This agent will help Zava:
- **Improve decision making** with faster sales insights
- **Reduce stockouts** by identifying low inventory and enabling transfers
- **Standardize reporting** for store managers and head office teams
- **Support omnichannel operations** by coordinating inventory across stores and online

Throughout this workshop, you'll use the AI Toolkit and Microsoft Foundry to build, test, and refine Cora's capabilities, learning how to create AI agents for real business scenarios.

Click **Next** to set up your Workshop environment and get started.