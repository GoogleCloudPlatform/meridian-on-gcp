# meridian on gcp

This repository provides cross organization teams - such as  product engineering, solutions, sales, professional services and partners - with an implementation path that drives the successful execution of a pilot engagement serving as a building block for Marketers and Sales teams interested in using Meridian Media Mix Modelling on Google Cloud.

[Meridian](https://developers.google.com/meridian) is an open-source MMM framework that enables advertisers to build and run their own in-house models to understand marketing ROI, channel performance, and budget optimization. Marketing mix modeling (MMM) uses aggregated data to measure marketing campaign impact across channels, informing budget planning and improving media effectiveness while maintaining user privacy. 

Meridian utilizes Bayesian causal inference, handles large-scale geo-level data, incorporates prior knowledge about media performance, and accounts for media saturation and lagged effects for accurate insights. Meridian helps you answer key questions such as:
* How did the marketing channels drive my revenue or other KPI?
* What was my marketing return on investment (ROI1)?
* How do I optimize my marketing budget allocation for the future?

The solution is targeted towards Agencies, Advertisers, Google Ads, Campaign Manager customers who want to optimize media performance and budget allocation across channels. It serves as blueprint, recommendation and opinionated implementation of a production-ready application using the open-source Meridian library. This packaged solution is different from the open source implementation, in many ways focusing on: automating the deployment of the meridian model and making it easy to run the optimization experiments; building a operational application with dashboards to enable the end-user to interpret the results and democratize access to the model insights in a privacy and safe manner; implementing production-ready pre-modelling and post-modelling pipelines leveraging best ml ops practices.

The tangible value to customers is to implement a ‘solution accelerator’ that enables marketers, advertisers to execute marketing campaigns better leveraging first party data foundation, predictive and generative AI capabilities to optimize budget allocation. Unlike other Google products or custom build from scratch solutions, this solution accelerator offers a fast time to value and a cost-efficient adoption of hardware accelerators.

Solution accelerators entail code samples that solve common business patterns to support customer adoption at scale, with positioning and a business case, released on Github.

## Pre-requisites

* Google Cloud Project
* Media channels metrics in a BigQuery dataset within that project
* An user with Google Cloud Project Owner role (or list of roles)
* L4 or A100 gpu quota availability for the chosen region to run the pipelines
* Marketing business owners to provide data and business insights

## Quick Installation ⏰

Want to quickly install and use it? Run this [installation notebook 📔](https://colab.sandbox.google.com/github/GoogleCloudPlatform/meridian-on-gcp/blob/main/notebooks/meridian_quick_install.ipynb) on Google Colaboratory and deploy Meridian in between 15-20 minutes.

## Installation Guide

Follow this [guide](infra/README.md).

