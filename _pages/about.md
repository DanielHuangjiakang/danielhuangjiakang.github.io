---
permalink: /
title: "Jiakang Huang"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

<div class="home-page">
  <a id="about"></a>
  <section class="home-section home-section--intro">
    <p class="home-page__lead">
      I am <strong>Jiakang (Daniel) Huang</strong>, a Bachelor of Science student in <strong>Computer Science</strong> at the
      <strong>University of British Columbia</strong>. My research interests lie at the intersection of AI systems, AI agents,
      LLM compilers, distributed databases, and memory systems.
    </p>
    <p>
      I am currently working with <strong>Henry Chan</strong>, a Principal Software Engineer at <strong>Huawei Canada's Field Lab</strong>,
      where I contribute to the development of GaussDB's shared-nothing distributed database system and its vector database branch.
      In parallel, I am conducting independent research on <strong>graph fusion strategies in PyTorch Inductor</strong>. During my
      third year, I also served as an undergraduate teaching assistant for <strong>CPSC 213</strong>, supporting students in
      computer systems and low-level programming.
    </p>
    <p>
      My long-term goal is to build sustainable, self-evolving <strong>AI agents</strong> that can continuously learn, adapt, and
      improve. My research path reflects this vision: my first paper focuses on <strong>natural language processing</strong>, aiming to help AI
      better understand human language; my second paper centers on AI infrastructure, targeting faster and more efficient systems
      for training and serving models; and my planned third paper will explore AI memory systems, enabling agents to develop
      stronger long-term memory.
    </p>
    <p>
      Beyond academics and research, I co-founded <strong>iMark</strong>, an AI bookmark assistant built on
      retrieval-augmented generation, memory-based personalization, and end-to-end product design. Outside of work, I enjoy
      basketball and CS:GO, where I once ranked in the top 5% globally.
    </p>
    <div class="home-tag-list">
      <span class="home-tag">AI Systems</span>
<span class="home-tag">LLM Compilers</span>
      <span class="home-tag">NLP</span>
      <span class="home-tag">Distributed Databases</span>
      <span class="home-tag">Memory Systems</span>
    </div>
  </section>

<a id="education"></a>

  <section class="home-section">
    <h2 class="home-section__title"><i class="fas fa-graduation-cap" aria-hidden="true"></i>Education</h2>
    <div class="home-card">
      <div class="home-card__body">
        <div class="home-card__logo-wrap"><img class="home-card__logo" src="/images/ubc-logo.png" alt="UBC" style="width:68px;height:68px;"></div>
        <div class="home-card__main">
          <h3>University of British Columbia</h3>
          <p class="home-card__detail">Bachelor of Science in Computer Science</p>
        </div>
      </div>
      <div class="home-card__period">Sep. 2022 - Dec. 2026</div>
    </div>
    <div class="home-card">
      <div class="home-card__body">
        <div class="home-card__logo-wrap"><img class="home-card__logo" src="/images/pku_logo.png" alt="PKU"></div>
        <div class="home-card__main">
          <h3>Peking University</h3>
          <p class="home-card__detail">Exchange Student, Computer Engineering</p>
        </div>
      </div>
      <div class="home-card__period">May 2024 - Sep. 2024</div>
    </div>
    <div class="home-card">
      <div class="home-card__body">
        <div class="home-card__logo-wrap"><img class="home-card__logo" src="/images/chengdu_fls_logo.png" alt="CFLS"></div>
        <div class="home-card__main">
          <h3>Chengdu Foreign Languages School</h3>
          <p class="home-card__detail">High School</p>
        </div>
      </div>
      <div class="home-card__period">Sep. 2019 - Jun. 2022</div>
    </div>
  </section>

<a id="experience"></a>

  <section class="home-section">
    <h2 class="home-section__title"><i class="fas fa-briefcase" aria-hidden="true"></i>Experience</h2>

    <div class="home-card">
      <div class="home-card__body">
        <div class="home-card__logo-wrap"><img class="home-card__logo" src="/images/Utoronto_logo.png" alt="UofT" style="width:68px;height:68px;"></div>
        <div class="home-card__main">
          <h3>University of Toronto</h3>
          <p class="home-card__detail">Research Assistant – Graph Fusion Optimization</p>
        </div>
      </div>
      <div class="home-card__period">Jan. 2026 - Current</div>
    </div>

    <div class="home-card">
      <div class="home-card__body">
        <div class="home-card__logo-wrap"><img class="home-card__logo" src="/images/imark_logo.png" alt="iMark" style="width:80px;height:80px;"></div>
        <div class="home-card__main">
          <h3>iMark</h3>
          <p class="home-card__detail">Co-founder & AI Technical Lead</p>
        </div>
      </div>
      <div class="home-card__period">Sep. 2025 - Current</div>
    </div>

    <div class="home-card">
      <div class="home-card__body">
        <div class="home-card__logo-wrap"><img class="home-card__logo" src="/images/huawei_logo.png" alt="GaussDB"></div>
        <div class="home-card__main">
          <h3>Huawei Canada</h3>
          <p class="home-card__detail">Software Engineering Co-op</p>
        </div>
      </div>
      <div class="home-card__period">Jan. 2025 – Dec. 2025</div>
    </div>

    <div class="home-card">
      <div class="home-card__body">
        <div class="home-card__logo-wrap"><img class="home-card__logo" src="/images/ubc-logo.png" alt="UBC" style="width:68px;height:68px;"></div>
        <div class="home-card__main">
          <h3>UBC Department of Computer Science</h3>
          <p class="home-card__detail">Undergraduate Teaching Assistant</p>
        </div>
      </div>
      <div class="home-card__period">Sep. 2024 – Dec. 2024</div>
    </div>

  </section>

<a id="publications"></a>

  <!-- <section class="home-section">
    <h2 class="home-section__title"><i class="fas fa-file-lines" aria-hidden="true"></i>Publications</h2>
    {% assign displayed_publications = 0 %}
    <div class="home-post-grid">
      {% for post in site.publications reversed %}
        {% if displayed_publications < 3 %}
          {% assign displayed_publications = displayed_publications | plus: 1 %}
          <article class="home-post-card">
            <p class="home-post-card__date">{{ post.date | date: "%B %d, %Y" }}</p>
            <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
            {% if post.venue %}<p><strong>{{ post.venue }}</strong></p>{% endif %}
            {% if post.excerpt %}<p>{{ post.excerpt | strip_html | strip_newlines | truncate: 220 }}</p>{% endif %}
          </article>
        {% endif %}
      {% endfor %}

      {% if displayed_publications == 0 %}
        <article class="home-post-card home-post-card--empty">
          <p class="home-post-card__date">No publications added yet</p>
          <h3>Publications</h3>
          <p>This section is ready for papers, preprints, or project write-ups.</p>
        </article>
      {% endif %}
    </div>
  </section> -->

<a id="blog"></a>

  <section class="home-section">
    <h2 class="home-section__title"><i class="fas fa-pen-nib" aria-hidden="true"></i>Blog</h2>
    {% assign displayed_posts = 0 %}
    <div class="home-post-grid">
      {% for post in site.posts %}
        {% unless post.title contains 'Blog Post number' or post.title == 'Future Blog Post' %}
          {% if displayed_posts < 3 %}
            {% assign displayed_posts = displayed_posts | plus: 1 %}
            <article class="home-post-card">
              <p class="home-post-card__date">{{ post.date | date: "%B %d, %Y" }}</p>
              <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
              <p>{{ post.excerpt | strip_html | strip_newlines | truncate: 180 }}</p>
            </article>
          {% endif %}
        {% endunless %}
      {% endfor %}
    </div>
  </section>
</div>
