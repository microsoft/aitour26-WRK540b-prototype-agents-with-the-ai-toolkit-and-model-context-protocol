## Source Code
Place source code you're sharing in your session in this folder!

## Setup Instructions

After cloning the repository and creating/activating your virtual environment:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.lock.txt
   ```
   
   Or if you need to update dependencies:
   ```bash
   pip install -r requirements.in
   ```

2. **Install shared package (if not already installed):**
   ```bash
   pip install -e ./shared
   ```

The shared package (`zava_shop_shared`) contains common utilities used across MCP servers including configuration, database providers, and models.

## Instructions
Follow the steps in the [docs folder](../docs) to get started.
