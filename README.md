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

Every post needs this front matter so search engines, social previews, and `llms.txt` work:

```yaml
---
layout: post
title: "Post Title"
date: 2026-11-01
categories: ML                # part of the URL, never change it after publishing
topic: Software Engineering   # one of the topics on /topics/
tags: [documentation, llm]    # 3 to 6 lowercase, hyphenated tags
description: "One sentence, 120 to 160 characters, stating what the post teaches or claims."
image: /assets/images/social/title.png
math: true                    # only if the post uses LaTeX
mermaid: true                 # only if the post uses <div class="mermaid">
series: Series Name           # only for multi-part posts
---
```

Topics in use: Machine Learning, GPU Programming, Signal Processing, Software Engineering, Opinion, Introduction.

After writing a post:

1. Add its metadata to `META` in `scripts/add_front_matter.py` and run it, or write the front matter by hand.
2. Run `python scripts/make_social_cards.py` to generate the 1200x630 preview image.
3. For any PNG over 300 KB, run `python scripts/optimize_images.py path/to/image.png` and link the `.webp` in the post.
4. Give every image descriptive alt text. Add `{: loading="lazy"}` after every image except the first.
5. After the post is live, run `scripts/indexnow.sh <post-url>` to notify Bing and other IndexNow engines.

Do not change a published post's `categories`, `date`, or file slug. All three are part of the URL.

### Search and AI discoverability

- `robots.txt`: allows search engines, AI search crawlers, and AI training crawlers.
- `llms.txt`: plain-text index of every post for LLMs and agents, generated from post front matter.
- `/topics/` and `/tags/`: index pages generated from `topic` and `tags`.
- `_layouts/post.html`: adds topic and tag links, series navigation, related posts, and a link to the markdown source.
- `_includes/head.html`: `jekyll-seo-tag` handles title, description, Open Graph, Twitter, and `BlogPosting` JSON-LD. The file adds `Blog` JSON-LD on the home page, `ProfilePage` on the about page, and article tags. MathJax and Mermaid load only on posts that set `math` or `mermaid`.
- `4796d92e7308419356ccae94fea9d4be.txt`: IndexNow key file. Keep it at the site root.

### Deployment

This site automatically deploys to GitHub Pages when changes are pushed to the main branch. No manual deployment is required.

### Useful Commands

```bash
# Install dependencies
docker-compose run --rm bundle

# Start development server
docker-compose up jekyll

# Build site for production (optional)
docker-compose run --rm jekyll jekyll build

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