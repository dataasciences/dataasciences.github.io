---
title: "Development Practice Smells to Watch out for"
excerpt: "Code Smells & Ways to Handle them"
categories:
  - Software Development
  - Debugging
tags:
  - Debugging
  - Programming
sidebar:
  - nav: docs
classes: wide
---

Recently, in a Tech town hall an interesting topic was discussed. In the age of AI, when code is automated (or atleast to begin with), what are some development practice smells to watch out for.  

Because here is the thing, in the world of software development, not all problems announce themselves with errors or glaring bugs. Some creep in quietly, and decisions that seem harmless at first slowly erode code quality, team efficiency, and long-term maintainability.
These are development practice smells, warning signs that your processes might be leading you toward technical debt and delivery pain. Much like code smells, they don’t guarantee failure, but they do hint at trouble ahead, if you do not know how to spot them & fix them at the right time.

> <span style="font-size:1em;"> Code smells are warning signs in your code that hint at deeper issues. These aren't errors and the code will still work, but they can make future development harder and increase the risk of bugs. </span>
{: .notice--info}

On that note, here are some ways to handle them;

## 1. Use Well-named tags to track the code version that you plan to deploy.

   * Use meaningful and descriptive tags (e.g., v1.2.0-prod) to clearly mark the code version that is deployed to a specific environment. This helps you easily trace back to the exact version, roll back if needed, and avoid ambiguity when communicating across teams.

## 2. Accurate Git configuration

   * Setting your Git username and email correctly ensures that commits are attributed to the right person. This not only makes the history more readable but also allows developers to get credit for their work. If you want to understand the basics of git and the major operations in git, you can [read them here](https://www.softwaremusings.dev/Git-Intro/).

## 3. Repository hooks enforce good Git configurations.

   * This is to ensure your repository follows a baseline of rules. Git hooks can enforce good practices during commit or push (e.g., rejecting code with wrong author info or missing commit message formats). They reduce human error and ensure the repository always follows a baseline of rules.

## 4. A well-thought-out branching strategy makes it easy to deliver change in all your environments, from dev to prod.

   * A clear branching model (e.g., main, dev, feature branches) avoids confusion and aligns across teams. It helps deliver changes in the correct order across environments and improves release reliability.

## 5. A Sensible merge strategy prevents code changes from getting lost.

   * Choosing a merge strategy (e.g., “squash and merge” or “rebase and merge”) ensures that code history remains clean and makes it harder for changes to disappear in large feature branches. This directly affects debugging and traceability.

## 6. Make your repositories world-readable.

   * When repos are readable by everybody in the organisation, you promote transparency and reuse. Teams can learn from each other’s code and avoid reinventing the wheel.

## 7. Use Sonar to get helpful feedback on your code.

   * Sonar (SonarQube/SonarCloud) or other similar Security analysis plugins automatically highlight code smells, anti-patterns, and potential bugs. It acts like a code reviewer that flags issues consistently and early.

## 8. Enable minimum successful build merge checks for your repositories.

   * Require that all merges pass CI tests before being merged into important branches. This ensures the branch is always in a deployable state and that broken code isn't introduced.

## 9. A well-written README provides a clear explanation of your code.

   * A clear README helps others understand why the project exists, how to set it up, and how to use it. It becomes the first entry point for collaborators or new team members.

## 10. Give actionable feedback on pull requests.

  * PR reviews should give concrete and constructive feedback (e.g., “consider splitting this into two methods” rather than “this is bad”). This helps others learn and results in better quality contributions.

## 11. A carefully crafted .gitignore keeps unwanted files out of your source code repositories.

  * A properly crafted .gitignore avoids committing local build artifacts, IDE configuration files, and temporary files. This keeps the repository clean and prevents unnecessary merge conflicts.

## 12. Use Artifactory for your binaries, don't commit them to your source code repositories.

  * Binary files don’t belong in version control repos and unnecessarily inflate repository size. Instead, store them in a dedicated artifact manager like Artifactory and version them properly there.

Technical debt, such as code smells, can be a developer's nightmare. Imagine your solution in place with all functionalitie,s only to know that there is a lot of clean up to be done later on.
Instead, implementing all the best practices discussed above during the development phase can save us a lot of time, which can be spent elsewhere in the SDLC pipeline.
