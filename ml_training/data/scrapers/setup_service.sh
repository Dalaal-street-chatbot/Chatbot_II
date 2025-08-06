#!/bin/bash
#
# Setup script for Dalaal Street Financial Data Refresh Service
# This script installs and configures the systemd service for financial data refresh

set -e

# Constants
SERVICE_NAME="data-refresh"
SCRIPT_PATH="$(realpath $(dirname "$0"))"
PROJECT_ROOT="$(realpath "$SCRIPT_PATH/../../..")"
SERVICE_FILE="$SCRIPT_PATH/data_refresh.service"
SYSTEMD_DIR="/etc/systemd/system"
SYSTEMD_SERVICE_PATH="$SYSTEMD_DIR/data-refresh.service"

# Color outputs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Dalaal Street Financial Data Refresh Service Setup ===${NC}"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root (use sudo)${NC}"
  exit 1
fi

# Function for installation
install_service() {
    echo -e "${BLUE}Installing data refresh service...${NC}"
    
    # Create logs directory if it doesn't exist
    mkdir -p "$SCRIPT_PATH/logs"
    chmod 755 "$SCRIPT_PATH/logs"
    
    # Make sure the script is executable
    chmod +x "$SCRIPT_PATH/data_refresh_scheduler.py"
    
    # Install required system packages
    echo -e "${YELLOW}Installing required system packages...${NC}"
    apt-get update
    apt-get install -y python3-psutil
    
    # Copy service file to systemd directory
    echo -e "${YELLOW}Copying service file to systemd...${NC}"
    cp "$SERVICE_FILE" "$SYSTEMD_SERVICE_PATH"
    chmod 644 "$SYSTEMD_SERVICE_PATH"
    
    # Update the WorkingDirectory in the service file
    echo -e "${YELLOW}Updating service configuration...${NC}"
    sed -i "s|WorkingDirectory=.*|WorkingDirectory=$PROJECT_ROOT|g" "$SYSTEMD_SERVICE_PATH"
    sed -i "s|ExecStart=.*|ExecStart=/usr/bin/python3 $SCRIPT_PATH/data_refresh_scheduler.py --daemon --log-level INFO --refresh-all|g" "$SYSTEMD_SERVICE_PATH"
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service to start on boot
    systemctl enable $SERVICE_NAME
    
    echo -e "${GREEN}Service installed successfully!${NC}"
    echo -e "${YELLOW}You can now start the service with: sudo systemctl start $SERVICE_NAME${NC}"
}

# Function to start service
start_service() {
    echo -e "${BLUE}Starting data refresh service...${NC}"
    systemctl start $SERVICE_NAME
    systemctl status $SERVICE_NAME
}

# Function to stop service
stop_service() {
    echo -e "${BLUE}Stopping data refresh service...${NC}"
    systemctl stop $SERVICE_NAME
    systemctl status $SERVICE_NAME
}

# Function to check status
check_status() {
    echo -e "${BLUE}Checking data refresh service status...${NC}"
    systemctl status $SERVICE_NAME
}

# Function to view logs
view_logs() {
    echo -e "${BLUE}Viewing service logs...${NC}"
    journalctl -u $SERVICE_NAME -n 50 --no-pager
    
    echo
    echo -e "${YELLOW}For more detailed logs, check:${NC}"
    echo -e "$SCRIPT_PATH/logs/data_refresh.log"
}

# Parse command line arguments
case "$1" in
    install)
        install_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        sleep 2
        start_service
        ;;
    status)
        check_status
        ;;
    logs)
        view_logs
        ;;
    *)
        echo -e "${BLUE}Dalaal Street Financial Data Refresh Service Setup${NC}"
        echo
        echo -e "Usage: sudo $0 {install|start|stop|restart|status|logs}"
        echo
        echo -e "  install    Install the service"
        echo -e "  start      Start the service"
        echo -e "  stop       Stop the service"
        echo -e "  restart    Restart the service"
        echo -e "  status     Check service status"
        echo -e "  logs       View service logs"
        exit 1
esac

exit 0
