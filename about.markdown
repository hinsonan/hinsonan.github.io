---
layout: page
title: About
permalink: /about/
---

<style>
  .about-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem 1rem;
  }
  
  .about-header {
    text-align: center;
    margin-bottom: 3rem;
  }
  
  .profile-image {
    width: 200px;
    height: 200px;
    border-radius: 50%;
    margin: 0 auto 2rem;
    display: block;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
  
  .about-intro {
    font-size: 1.25rem;
    line-height: 1.8;
    color: #555;
    margin-bottom: 2rem;
    text-align: center;
  }
  
  .about-section {
    margin-bottom: 3rem;
  }
  
  .about-section h2 {
    color: #333;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
  }
  
  .skills-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }
  
  .skill-category h3 {
    color: #666;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
  }
  
  .skill-category ul {
    list-style: none;
    padding-left: 0;
  }
  
  .skill-category li {
    padding: 0.25rem 0;
    color: #777;
  }
  
  .contact-info {
    padding: 2rem 0;
    text-align: center;
    border-top: 2px solid #e0e0e0;
  }
  
  body.dark-mode .contact-info {
    border-top-color: #444;
  }
  
  .social-links {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 0;
  }
  
  .social-links a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background-color: #333;
    color: white;
    text-decoration: none;
    transition: transform 0.3s ease, background-color 0.3s ease;
  }
  
  body.dark-mode .social-links a {
    background-color: #666;
  }
  
  .social-links a:hover {
    transform: translateY(-3px);
  }
  
  .social-links a.instagram:hover {
    background: linear-gradient(45deg, #f09433 0%,#e6683c 25%,#dc2743 50%,#cc2366 75%,#bc1888 100%);
  }
  
  .social-links a.linkedin:hover {
    background-color: #0077b5;
  }
  
  .social-links a.twitter:hover {
    background-color: #000000;
  }
  
  .social-links svg {
    width: 24px;
    height: 24px;
    fill: currentColor;
  }
  
  @media (max-width: 600px) {
    .profile-image {
      width: 150px;
      height: 150px;
    }
    
    .about-intro {
      font-size: 1.1rem;
    }
  }
</style>

<div class="about-container">
  <div class="about-header">
    <img src="/assets/images/robot.png" alt="Profile Photo" class="profile-image">
    <p class="about-intro">
      This industry is a mess and I'm here to try and navigate through the chaos 
    </p>
  </div>

  <div class="about-section">
    <h2>Who is this guy?</h2>
    <p>
      I have been working in the field for a while now. The reason for this blog is multi-fold (multi-modal). I want to start getting my ideas out there in a more formal way and better my delivery skills. I see a lot of developers struggling in this field or people that want to increase their knowledge in ML or Software Engineering.
    </p>
    <p>
      I have the unique perspective of someone who started in college as a web dev, then moved into radar simulation work and then into ML. I have spent the majority of my career doing ML and no I don't mean prompt engineering. I was in this field before LLMs. I have a large amount of both Software and ML experience. I have worked on many projects and lead projects in multiple ways. I say this to lay the ground work into how I view problems. I am not a dork who lives in academia and can't solve problems in professional ways. 
    </p>
    <p>
      Machine Learning is hard and too many academics release these white papers along with awful repos that barely work and in many cases are broken beyond repair. It makes you question all their research. Software is hard because of many reasons. Todays systems grow in complexity quickly and it's our job to slay that complexity like the doom slayer. 
    </p>
    <p>
      I don't have all the answers but gosh darn it I got a few of them. Join me as we dive into the world of Machine Learning and Software Development 
    </p>
  </div>

  <div class="contact-info">
    <div class="social-links">
      <a href="https://www.instagram.com/hinsonan" target="_blank" rel="noopener noreferrer" class="instagram" aria-label="Instagram">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zM5.838 12a6.162 6.162 0 1112.324 0 6.162 6.162 0 01-12.324 0zM12 16a4 4 0 110-8 4 4 0 010 8zm4.965-10.405a1.44 1.44 0 112.881.001 1.44 1.44 0 01-2.881-.001z"/>
        </svg>
      </a>
      <a href="https://www.linkedin.com/in/hinsonan" target="_blank" rel="noopener noreferrer" class="linkedin" aria-label="LinkedIn">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
        </svg>
      </a>
      <a href="https://x.com/hinsonan" target="_blank" rel="noopener noreferrer" class="twitter" aria-label="X (Twitter)">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
        </svg>
      </a>
    </div>
  </div>
</div>
