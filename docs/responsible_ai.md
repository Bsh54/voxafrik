# Responsible AI

A system that decides what to flag holds real power over speech. These principles
are built into VoxAfrik, not added afterwards.

## An insult is not hate speech

VoxAfrik uses three classes, not a binary. ABUSIVE (a personal insult) is routed
differently from HATE (an attack on a group for what its members are). A tool that
conflates them censors ordinary argument and humour.

## The human decides

The system triages and explains; it never removes content on its own. Every alert
carries the original text, the English translation, the class, the confidence and
the words that triggered it, so a moderator can overrule the model.

## Report per language, never an average

Accuracy differs by language (strong on Swahili and Igbo, weaker on Hausa and
Yoruba). A global average would hide that some communities are served worse than
others. Metrics are published per language and treated as a defect to fix, not a
fact to accept.

## Doubt favours expression

Low-confidence cases go to human review, never to automatic blocking. A missed
hateful post and a wrongly blocked legitimate post are different harms; the system
is biased toward recall while keeping a human in the loop.

## No personal data

VoxAfrik uses only public, research-grade datasets with no personal or health
information, consistent with the LinuxONE Community Cloud terms.
