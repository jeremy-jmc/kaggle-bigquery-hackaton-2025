---
description: 'Kaggle BigQuery Hackaton 2025'
tools: []
---

# Kaggle Competition: BigQuery AI - Building the Future of Data Build AI solutions with BigQuery

## Overview

We challenge you to go beyond traditional analytics and build groundbreaking solutions using BigQuery's cutting-edge AI capabilities. This is your opportunity to solve real-world business problems using BigQuery’s Generative AI, Vector Search, and Multimodal capabilities. / Use BigQuery’s Generative AI features to solve real-world business problems. BigQuery has SQL and Python (BigFrames) support for calling LLMs and this track challenges you to leverage that support and BigQuery’s analytical prowess to understand a business problem and then build a prototype for a business application or workflow.

## BigQuery AI Methods

- Generative AI in SQL:
    - `ML.GENERATE_TEXT`: The classic function for large-scale text generation.
    - `AI.GENERATE`: Generate free-form text or structured data based on a schema from a prompt.
    - `AI.GENERATE_BOOL`: Get a simple True/False answer about your data.
    - `AI.GENERATE_DOUBLE`: Extract a specific decimal number from text.
    - `AI.GENERATE_INT`: Extract a specific whole number from text.
    - `AI.GENERATE_TABLE`: Create a structured table of data from a single prompt.
    - `AI.FORECAST`: Predict future values for time-series data with a single function.

- Vector Search in SQL:
    - `ML.GENERATE_EMBEDDING`: Transform your data (text, images) into meaningful vector representations.
    - `VECTOR_SEARCH`: The core function to find items based on meaning, not just keywords. Can be used with or without a vector index.
    - `CREATE VECTOR INDEX`: Build an index for speeding up similarity search on larger tables (1 million rows or above)

## Description

Companies are sitting on piles of data, including chat logs, PDFs, screenshots, and recordings, but they can’t do much with it. Existing tools are typically built for just one data format, or they require too much manual work. This makes it hard to find patterns, generate content, or even answer basic questions.

Your task is to build a working prototype that uses BigQuery’s AI capabilities to process unstructured or mixed-format data. That might mean pulling up similar documents from a giant text archive, creating summaries on the fly, or stitching together insights from messy, mixed data. Whatever you build, the idea is to solve a real problem using tools that feel like an extension of SQL, not a separate system.

You’ll have access to public datasets, and you’re welcome to bring your own as long as they’re publicly available. The goal is to demonstrate how AI within BigQuery can address real-world problems that go beyond rows and columns.

Whether you submit a demo, a notebook, or a walkthrough, we want to see how you utilize these tools to make sense of data that is often overlooked.

## Submission Requirements

The challenge offers three approaches. You must use at least one approach, but you may use two or all three. Submissions are only eligible for one prize. Inspiration is provided for each approach, but should be considered as the type of project that would be considered a great application.

### Approach 1: The AI Architect 🧠

Your Mission: Use BigQuery's built-in generative AI to architect intelligent business applications and workflows. Build solutions that can generate creative content, summarize complex information, or even forecast future trends directly within your data warehouse.

**Inspiration:**

- Build a Hyper-Personalized Marketing Engine: Generate unique marketing emails for every customer based on their individual purchase history and preferences.
- Create an Executive "Insight" Dashboard: Develop a dashboard that automatically ingests raw support call logs and transforms them into summarized, categorized, and actionable business insights.

### Approach 2: The Semantic Detective 🕵️‍♀️

Your Mission: Go beyond keyword matching and uncover deep, semantic relationships in your data using BigQuery's native vector search. Build systems that understand meaning and context to find similar items, concepts, or issues with incredible accuracy.

**Inspiration:**

- Develop an Intelligent Triage Bot: Instantly find historical support tickets that are semantically similar to a new, incoming ticket to speed up resolution time. The bot may also recommend a solution based on past ticket resolutions.
- Design a "Smart Substitute" Recommender: Suggest ideal product substitutes based on a deep understanding of product attributes, not just shared tags or categories.

### Approach 3: The Multimodal Pioneer 🖼️

Your Mission: Break the barriers between structured and unstructured data using BigQuery's multimodal capabilities. Combine numerical and categorical data with images, documents, and other rich media to unlock insights that are impossible to find in siloed datasets.

**Inspiration:**

- Revolutionize Real Estate Valuation: Improve property price predictions by fusing structured data (e.g., sqft, # of bedrooms) with unstructured data from street-level and satellite imagery.
- Build an Automated Quality Control Agent: Detect discrepancies between a product's listed specifications, its marketing description, and its actual product image.

# MY PROJECT IDEA DESCRIPTION

## APPROACH 1: Simple Recommender System using User and Item Profiles generated using BigQuery's GenAI capabilities

Create a RecSys based on User and Item Profiles generated using BigQuery's GenAI capabilities (`AI.GENERATE`) in which starting from creating user and item embeddings (`ML.GENERATE_EMBEDDING`) and then apply vector search (`VECTOR_SEARCH`) to generate recommendations.
- **Pros:** Simpler but interesting approach
- **Cons:** Too expensive to run on large datasets.


## APPROACH 2: Unveiling patterns in food and user LLM profiles using BigQuery AI to enhance recommendation system

Based on the textual LLM-generated User and Item profiles (`AI.GENERATE`), improve the prediction accuracy of a explainable RecSys (XGBoost Ranker) using additional features extracted from the profiles. The LLM gives ideas/hypothesis of what features are missing in the current dataset and can be extracted from the profiles. Then, the human implement it to show how BigQueryAI can help in the data understanding and feature engineering process to improve the model performance. We could also generate hypothetical explanations based on SHAP values and LLM (`AI.GENERATE`) to explain why a certain item is recommended to a user based on the profiles.

- **Pros:** Use less tokens than the previous approach, more explainable. Pretty novel approach and interesting for companies developing RecSys.
- **Cons:** More complex to implement, need to be careful about data leakage.


## APPROACH 3: Using Both Previous Approaches

We could also combine both approaches to create a hybrid recommender system that uses both collaborative filtering (`VECTOR_SEARCH`) and content-based filtering (XGBoost Ranker with features extracted from LLM profiles - `AI.GENERATE`). The idea is to reduce the number of tokens, limiting the calls of `VECTOR_SEARCH` to a shortlist of items (users with low HITRate@5 generated by XGBoost).



## References: 
- [Profile Generation with LLMs: Understanding consumers, merchants, and items](https://careersatdoordash.com/blog/doordash-profile-generation-llms-understanding-consumers-merchants-and-items/)
- [MealRec+: A Meal Recommendation Dataset with Meal-Course Affiliation for Personalization and Healthiness](https://github.com/WUT-IDEA/MealRecPlus)
- [MealRec: A Meal Recommendation Dataset](https://arxiv.org/abs/2205.12133)
- [Dataset foodRecSys-V1](https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1)
- [Evaluating Podcast Recommendations with Profile-Aware LLM-as-a-Judge](https://arxiv.org/abs/2508.08777)

