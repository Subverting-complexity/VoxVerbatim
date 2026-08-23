"""Turning a recording into a transcript by combining several services.

No single speech-to-text service is the best at everything. One is good at
hearing unusual names, another at saying exactly when each word was spoken,
another at telling the speakers apart. So this package runs several of them
over the same recording and works out one answer from what they all said.

The idea that shapes everything here is that a transcript answers three
separate questions, and that the best answer to each can come from a
different service:

* What was said.
* When it was said.
* Who said it.

They are therefore kept apart. A word can take its spelling from one
service, its timing from another and its speaker from a third, and that is
normal rather than a special case. Every one of those three carries a note
of where it came from and how sure we are of it.

The second idea is that being unsure is a real answer. Where the evidence
does not settle a question, the transcript says so and the word waits for a
person to decide, rather than the application choosing whichever candidate
looked most plausible and hiding the doubt.
"""
