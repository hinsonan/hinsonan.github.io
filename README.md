# hinsonan.github.io
Machine Learning and Software Development Blog

A Jekyll-powered blog hosted on GitHub Pages, covering topics in machine learning, software development, and technical tutorials.

## Development Setup

This project uses Docker Compose to provide a consistent Jekyll development environment without requiring local Ruby installation.

### Prerequisites

- Docker
- Docker Compose

### Initial Site Creation

If you're setting up the site from scratch:

```bash
# Create new Jekyll site
docker-compose run --rm init

# Install dependencies
docker-compose run --rm bundle
```

### Development Workflow

#### Starting the Development Server

```bash
# Start Jekyll development server with live reload
docker-compose up jekyll
```

The site will be available at:
- **Local site**: http://localhost:4000
- **Live reload**: Automatically refreshes when files change

#### Installing Dependencies

```bash
# Install gems after updating Gemfile
docker-compose run --rm bundle

# Update gems to latest versions
docker-compose run --rm bundle bundle update
```

#### Stopping the Server

```bash
# Stop the development server
Ctrl+C (or Cmd+C on Mac)

# Or stop all services
docker-compose down
```

### Docker Compose Services

- **`jekyll`**: Main development server with live reload
- **`bundle`**: Runs bundle commands for dependency management  
- **`init`**: Creates new Jekyll site (used once during setup)

### Writing Posts

Create new blog posts in the `_posts/` directory using the format:
```
YYYY-MM-DD-title.md
```

Example:
```
_posts/2025-07-11-my-first-post.md
```

### Deployment

This site automatically deploys to GitHub Pages when changes are pushed to the main branch. No manual deployment is required.

### Useful Commands

```bash
# Install dependencies
docker-compose run --rm bundle

# Start development server
docker-compose up jekyll

# Build site for production (optional)
docker-compose run --rm jekyll bundle exec jekyll build

# Access container shell for debugging
docker-compose run --rm jekyll sh
```

## Project Structure

```
├── _posts/              # Blog posts
├── _config.yml          # Jekyll configuration
├── _layouts/            # Page templates
├── _includes/           # Reusable components
├── assets/              # CSS, JS, images
├── docker-compose.yml   # Docker development setup
├── Gemfile              # Ruby dependencies
└── index.md             # Homepage
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally using `docker-compose up jekyll`
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).