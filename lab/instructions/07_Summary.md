# Summary

In this lab, you learned how to:

- Explore and compare models to select the best fit for your business scenario
- Augment models with prompts and data to get more accurate and grounded responses
- Prototype an internal-facing agent by combining models and instructions with tools via MCP (Model Context Protocol)
- Extract the agent's code for further customization and deployment

Across this process, you also gained hands-on experience with the AI toolkit in Visual Studio Code, which is designed to streamline the development of AI-powered applications.

## Next steps

As you move forward in your development journey with AI agents, and consider deploying them in production environments, there's a few important considerations to keep in mind:

- **Azure hosted models**: For production scenarios, it's advisable to use Azure-hosted models. These models offer better performance, reliability, and compliance with enterprise standards. You can explore the available catalog in [Microsoft Foundry Models](https://ai.azure.com/catalog).
- **Evaluation**: Before deploying an agent, it's crucial to evaluate its performance thoroughly. This includes testing its responses for accuracy, relevance, and safety. Consider using a mix of automated tests and human evaluations. You can learn more about agent evaluation in the [official documentation](https://code.visualstudio.com/docs/intelligentapps/evaluation).
- **Deployment**: When deploying your agent, consider the infrastructure and platform that best suits your needs. An application similar to the one you prototyped in this lab - which includes a Python application based on Microsoft Agent Framework, a Microsoft Foundry hosted model and MCP servers - for example can be deployed using Azure Container Apps or Azure Kubernetes Service (AKS). These services provide scalability and reliability for production workloads.
- **Monitoring**: Once deployed, continuously monitor the agent's performance in real-world scenarios. This helps in identifying any issues or areas for improvement. Set up logging and alerting mechanisms to track the agent's behavior and performance metrics. The observability features in Microsoft Foundry can be very helpful for this purpose. Discover more in the [official documentation](https://learn.microsoft.com/azure/ai-foundry/how-to/monitor-applications).
- **Continuous improvement**: AI agents can always be improved. Gather user feedback and analyze the agent's interactions to identify areas for enhancement. Regularly update the agent's model, prompts, and tools to keep it effective and relevant.

## Try this at home

This lab is available for you to revisit at your own pace. You can find the complete lab instructions and resources in the official [GitHub repository](https://github.com/microsoft/aitour26-WRK542-prototype-agents-with-the-ai-toolkit-and-model-context-protocol).