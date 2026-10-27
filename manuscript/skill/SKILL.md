---
name: edit-for-clarity
description: Edit a piece of English text so it is easier for a human reader to understand, by removing six sentence patterns that research links to reading difficulty and that AI models overuse. Use this skill whenever the user asks to make text clearer, simpler, more readable, more human, less "AI-sounding", or asks to edit, polish, or proofread prose for a human audience (emails, docs, READMEs, PR descriptions, commit messages, reports, essays, chat replies). Also use it to review your own drafts before sending them, even if the user did not ask.
---

# Edit for Clarity

Make text easier for a human to read without changing what it says.

## Source

This skill is based on:

> Thomas Hikaru Clark, Matias B. Martin, and Kyle Mahowald (2026).
> *Why is AI-generated language so hard to understand?*
> arXiv:2610.29162. https://arxiv.org/abs/2610.29162

The paper measured AI-written text across chat, coding, and essay tasks and found that AI models use six sentence patterns more often than humans do and more often than needed to express the same meaning. Each pattern is known from psycholinguistics to slow readers down. The paper suggests this happens because models are trained to be compact, not to be understood. This skill does not reproduce the paper. It turns its findings into editing rules. Read the paper for the evidence.

## The rule in one sentence

Prefer the version a careful human writer would choose, even if it uses a few more words.

## The six patterns to fix

Scan every sentence for these. Fix each one you find. The patterns are listed from most to least common in AI text.

### 1. Rare word combinations

**What it is:** Two or three words placed together that readers have rarely or never seen together. Often a creative metaphor or a coined phrase.

**Why it hurts:** Readers predict words as they read. Unpredictable words cost extra effort.

**How to find it:** Ask: "Would I expect to see these two words next to each other in a normal book or conversation?" If not, flag it.

**How to fix:** Replace with the common, plain phrase that means the same thing.

| Before | After |
|---|---|
| The deadline is quietly eating your margin. | The deadline is reducing your margin. |
| This is the config-drift tax you pay later. | This is a problem that config drift causes later. |

### 2. Objects or ideas doing things only people do (inanimate subjects)

**What it is:** A non-living thing is the subject of a verb that normally needs a living subject. Examples of such verbs: wants, expects, forgets, swallows, bites, promises, hopes, misunderstands, assumes, complains.

**Why it hurts:** Readers use "is this subject alive?" as a cue to parse the sentence. A mismatch forces them to re-read.

**How to find it:** Look at the subject of each verb. If the subject is a function, file, variable, document, plan, number, or idea, check the verb. Would a person normally be the one doing this verb?

**How to fix:** Rewrite so the verb is one that things can do (returns, causes, contains, shows, fails), or make the person the subject.

| Before | After |
|---|---|
| `parse()` swallows the error and returns None. | `parse()` catches the error and returns None. |
| The spec expects you to remember the default. | The spec does not state the default, so you must remember it. |
| This test forgets about empty input. | This test does not cover empty input. |

Note: Some of these are fully standard English ("the function returns", "the file contains"). Only fix verbs that clearly need a living subject.

### 3. Nouns used as verbs (denominal verbs)

**What it is:** A noun turned into a verb, especially when it is unusual: "calendar it", "footnote this", "cushion the spike", "sandbox the process", "bucket the results".

**Why it hurts:** The reader first reads the word as a noun, then has to re-parse it as a verb.

**How to find it:** For each verb, ask: "Is this word normally a noun?" If yes, and the verb use is not completely standard (like "email me" or "google it"), flag it.

**How to fix:** Use a normal verb plus the noun: "put it on the calendar", "add a footnote", "soften the spike", "run the process in a sandbox", "sort the results into groups".

| Before | After |
|---|---|
| I'll ticket this for next sprint. | I'll create a ticket for this for next sprint. |
| High inventory cushions price spikes. | High inventory softens price spikes. |

### 4. Missing "that" (reduced object relative clauses)

**What it is:** A clause that describes a noun, where the word "that" or "which" has been dropped: "the bug *the old code* never hit", "the money *you* saved".

**Why it hurts:** Without "that", the reader may briefly read the two nouns as subject and object of one clause. The word "that" tells them a new clause is starting.

**How to find it:** Look for two nouns or pronouns right next to each other with no connecting word: "the file the script", "the feature users", "the error we".

**How to fix:** Insert "that" (or "which", "who", "whom" when more natural). If the sentence is still hard, restructure it (see pattern 6).

| Before | After |
|---|---|
| Three tests check behaviour the old code never reached. | Three tests check behaviour that the old code never reached. |
| The fix the reviewer asked for is in. | The fix that the reviewer asked for is in. |

Also apply this to "that" after verbs of thinking and saying when the following clause is long or could be misread: "I think the cache is stale" is fine; "I confirmed the file the script writes is stale" should become "I confirmed that the file the script writes is stale", and better still "I confirmed that the file written by the script is stale".

### 5. Stacked nouns

**What it is:** Three or more nouns in a row acting as one phrase: "dotfile backup path handling", "class-time shutdown requirement", "user session token refresh logic".

**Why it hurts:** The reader cannot tell which noun modifies which. "Dotfile backup path handling" could group in several ways.

**How to find it:** Count consecutive nouns (hyphenated compounds count as one). Three or more in a row: flag it.

**How to fix:** Unpack with prepositions ("of", "for", "in"), or split into two phrases, or drop nouns that add nothing.

| Before | After |
|---|---|
| Fixed the dotfile backup path handling. | Fixed how backup paths for dotfiles are handled. |
| The user session token refresh logic is broken. | The logic that refreshes session tokens for users is broken. |

Two nouns in a row ("error message", "test suite") are normal. Leave them.

### 6. Long gaps between related words (long dependencies)

**What it is:** The subject and its verb, or a verb and its object, are far apart because something long was inserted between them.

**Why it hurts:** The reader has to hold the first word in memory until the second arrives.

**How to find it:** Find the subject of the sentence. Count words until its main verb. More than about 8 words between them: flag it. Also flag long lists or asides placed between a verb and its object.

**How to fix:** Move the inserted material to the start or end of the sentence, or split into two sentences.

| Before | After |
|---|---|
| All five expression shapes, including attribute access, subscripts, calls, nested calls, and string literals, leave the cursor on the same token. | All five expression shapes leave the cursor on the same token. These are attribute access, subscripts, calls, nested calls, and string literals. |
| The guitar sitting in the corner because you are still deciding teaches you zero chords. | You learn zero chords from a guitar that sits in the corner while you decide. |

## Procedure

Follow these steps in order.

1. **Read the whole text once** without editing. Note who the reader is and what the text is for.
2. **Go sentence by sentence.** For each sentence, check the six patterns in order. Fix every instance you find.
3. **Check your fixes against the meaning.** The edited sentence must say exactly what the original said. If you are not sure what the original meant, do not guess. Keep it and add a note to the user.
4. **Check that you did not add new problems.** A fix for pattern 3 must not create pattern 5. Re-scan edited sentences.
5. **Output the result** in the format described below.

## What NOT to change

- **Meaning.** Never add, remove, or soften a claim.
- **Code.** Never edit anything inside backticks or code blocks. Function names, variables, file paths, commands, and error messages stay exactly as written.
- **Quotes.** Never edit text that the author is quoting from someone else.
- **Technical terms the reader knows.** If the audience is developers, "race condition" is fine. Do not replace standard terms with longer explanations.
- **Tone.** Keep casual text casual and formal text formal.
- **Structure.** Keep headings, lists, and paragraph breaks unless a sentence split requires a small change.
- **Correct sentences.** If a sentence has none of the six patterns, leave it alone. Do not rewrite for taste.

Adding words is fine. The goal is clarity, not brevity. A sentence that is 20% longer but reads in one pass is a good trade.

## Output format

Default: return the full edited text, then a short list of the changes made, grouped by pattern number. Example:

```
[edited text here]

---
Changes:
- Pattern 2 (inanimate subject): "the parser swallows" → "the parser catches"
- Pattern 4 (missing "that"): added "that" in 3 places
- Pattern 5 (stacked nouns): "session token refresh logic" → "logic that refreshes session tokens"
- Pattern 6 (long gap): split one 34-word sentence into two
```

If the user asks for edits in a file, edit the file in place and report the change list in chat.

If the user says "just the text" or similar, return only the edited text.

If the text has no instances of any pattern, say so and return it unchanged.

## Quick checklist

Use this when reviewing your own draft before sending.

- [ ] No word pairs a reader would find surprising
- [ ] No objects or ideas "wanting", "expecting", "forgetting", "biting"
- [ ] No nouns used as unusual verbs
- [ ] "that" is present where two nouns would otherwise collide
- [ ] No runs of three or more nouns
- [ ] Subject and verb are close together; long asides moved or split
- [ ] Meaning, code, quotes, and tone are unchanged