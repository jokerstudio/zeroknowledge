# Dev Container Configuration

This directory contains the **only supported Docker configuration** for running this project. We use VS Code/Cursor Dev Containers for a seamless development experience.

## What's Included

- **Python 3.13**: Latest stable Python version
- **All Project Dependencies**: Automatically installed from `requirements.txt`
- **Jupyter Support**: Full notebook support with VS Code/Cursor extensions
- **VS Code Extensions**: Python, Pylance, and Jupyter extensions pre-installed
- **Git**: Version control tools with Zsh/Oh My Zsh
- **Non-root User**: Runs as `vscode` user for better security
- **Automatic Port Forwarding**: Port 8888 automatically forwarded to localhost

## Files

- **`devcontainer.json`**: Main configuration file for the dev container
  - Defines extensions, settings, and port forwarding
  - Configures post-create commands
  - Sets up features like Git and common utilities

- **`Dockerfile`**: Docker image definition
  - Based on Python 3.13-slim
  - Installs system dependencies and Python packages
  - Creates non-root user

## How to Use

### Prerequisites
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Install [VS Code](https://code.visualstudio.com/) or [Cursor IDE](https://cursor.sh/)
3. Install the ["Dev Containers" extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Opening the Project

1. Open the project folder in VS Code/Cursor
2. When prompted, click **"Reopen in Container"**
   - Or use Command Palette (`Cmd/Ctrl+Shift+P`) → **"Dev Containers: Reopen in Container"**
3. Wait for the container to build (first time only, ~2-3 minutes)
4. You're ready to code! All dependencies are installed automatically.

### Port Forwarding

Port 8888 is automatically forwarded for Jupyter Notebook:
- Check the **"Ports"** tab in VS Code/Cursor (bottom panel)
- Access Jupyter at `http://localhost:8888`
- You'll receive a notification when the port is forwarded

### Working with Notebooks

**Option 1: Direct in VS Code/Cursor (Recommended)**
- Click any `.ipynb` file
- Select Python kernel when prompted
- Start coding!

**Option 2: Jupyter Notebook Server**
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```
Access at `http://localhost:8888` (port auto-forwarded)

### Rebuilding the Container

If you modify `Dockerfile` or `devcontainer.json`:
1. Command Palette (`Cmd/Ctrl+Shift+P`)
2. Select **"Dev Containers: Rebuild Container"**
   - Or use **"Rebuild Without Cache"** for a clean build

## Benefits

✅ **Consistency**: Same environment for all developers  
✅ **Isolation**: No conflicts with local Python installations  
✅ **Quick Setup**: New contributors start in minutes  
✅ **Reproducibility**: Locked Python version and dependencies  
✅ **No Manual Setup**: Everything is automated  
✅ **Port Forwarding**: Automatic, no configuration needed  
✅ **IDE Integration**: Full VS Code/Cursor features in container

## Troubleshooting

### Container won't build
- Ensure Docker Desktop is **running**
- Check Docker has enough resources: Docker Desktop → Settings → Resources (recommend 4GB+ RAM)
- Try rebuilding: **"Dev Containers: Rebuild Container Without Cache"**
- Check Docker logs for specific errors

### Port 8888 not accessible
1. Open the **"Ports"** tab (bottom panel)
2. Verify port 8888 is listed and status is "Forwarding"
3. If not, manually forward: Click "Forward a Port" → Enter `8888`
4. Check if something else is using port 8888:
   ```bash
   # Mac/Linux
   lsof -i :8888
   
   # Windows
   netstat -ano | findstr :8888
   ```

### Jupyter not working
- Check terminal output for errors
- Verify installation: `jupyter --version`
- Restart container: **"Dev Containers: Rebuild Container"**
- Check Python kernel is selected in notebooks

### Slow performance
- Allocate more resources to Docker Desktop (Settings → Resources)
- Close unused applications
- Restart Docker Desktop
- Consider using Docker volumes for large datasets (already configured)

### Changes not appearing
- Save your files (`Cmd/Ctrl+S`)
- Restart Jupyter kernel (in notebook)
- Check file isn't in `.dockerignore`
- Rebuild if you changed dependencies

## Why Dev Containers Only?

We've chosen to support **only Dev Containers** (not docker-compose, manual docker run, etc.) because:

1. **Seamless IDE Integration**: Full VS Code/Cursor features work in container
2. **Automatic Port Forwarding**: No manual configuration needed
3. **Better Developer Experience**: Extensions, settings, all pre-configured
4. **Easier Onboarding**: One-click setup for new contributors
5. **Industry Standard**: Widely adopted in modern development workflows

## Need Help?

- Check main [README.md](../README.md) for usage instructions
- See [VS Code Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- Check [Docker Desktop Documentation](https://docs.docker.com/desktop/)
