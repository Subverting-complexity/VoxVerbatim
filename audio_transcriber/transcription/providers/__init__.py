"""Adapters for the speech-to-text services.

Each service is wrapped in an adapter that hides everything specific to it:
its client library, its parameter names, the shape of its answer, its file
size limit and its errors. Above this line nothing knows or cares which
service produced a word.

That boundary is what lets a service be swapped, added or dropped without
the reconciliation rules changing, and it is why the specification insists
on it. Every adapter answers the same three questions in the same shape: it
declares what it can do, it accepts one request, and it returns words on the
canonical timeline or an error.
"""
