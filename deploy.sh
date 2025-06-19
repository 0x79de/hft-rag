#!/bin/bash

# HFT-RAG Deployment Script
set -e

echo "🚀 Starting HFT-RAG deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed. Please install docker-compose first."
        exit 1
    fi
    
    print_status "Dependencies check passed ✓"
}

# Build and test
build_and_test() {
    print_status "Building and testing application..."
    
    # Run tests
    if ! cargo test; then
        print_error "Tests failed. Deployment aborted."
        exit 1
    fi
    
    # Check code quality
    if ! cargo clippy --all-targets --all-features -- -D warnings; then
        print_error "Code quality checks failed. Deployment aborted."
        exit 1
    fi
    
    print_status "Build and test completed ✓"
}

# Deploy with docker-compose
deploy() {
    print_status "Deploying with docker-compose..."
    
    # Stop existing containers
    docker-compose down || true
    
    # Build and start services
    docker-compose up --build -d
    
    print_status "Deployment completed ✓"
}

# Wait for services to be healthy
wait_for_health() {
    print_status "Waiting for services to become healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps | grep -q "healthy"; then
            print_status "Services are healthy ✓"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts - Services not yet healthy, waiting..."
        sleep 10
        ((attempt++))
    done
    
    print_error "Services failed to become healthy within timeout"
    docker-compose logs
    exit 1
}

# Show status
show_status() {
    print_status "Deployment status:"
    docker-compose ps
    
    echo ""
    print_status "Service URLs:"
    echo "  • HFT-RAG API: http://localhost:8080"
    echo "  • Health Check: http://localhost:8080/health"
    echo "  • Qdrant API: http://localhost:6333"
    echo "  • Qdrant Dashboard: http://localhost:6333/dashboard"
    
    echo ""
    print_status "Useful commands:"
    echo "  • View logs: docker-compose logs -f"
    echo "  • Stop services: docker-compose down"
    echo "  • Restart: docker-compose restart"
    echo "  • Update: ./deploy.sh"
}

# Main execution
main() {
    case "${1:-deploy}" in
        "check")
            check_dependencies
            ;;
        "test")
            build_and_test
            ;;
        "deploy")
            check_dependencies
            build_and_test
            deploy
            wait_for_health
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            docker-compose logs -f
            ;;
        "stop")
            print_status "Stopping services..."
            docker-compose down
            print_status "Services stopped ✓"
            ;;
        *)
            echo "Usage: $0 [check|test|deploy|status|logs|stop]"
            echo ""
            echo "Commands:"
            echo "  check   - Check dependencies"
            echo "  test    - Run tests only"
            echo "  deploy  - Full deployment (default)"
            echo "  status  - Show deployment status"
            echo "  logs    - Show service logs"
            echo "  stop    - Stop all services"
            exit 1
            ;;
    esac
}

main "$@"