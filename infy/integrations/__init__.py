"""Optional integrations that let infy governance wrap third-party agent frameworks.

Each integration lazily imports its framework, so the zero-dependency core is untouched: you only
pull in a framework when you import its adapter (for example ``infy.integrations.smolagents``).
"""
