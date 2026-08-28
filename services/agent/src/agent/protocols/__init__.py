"""
different ways to call the agent. the agent created is identical (always instantiate the agent from core) but the 
way the response is returned back to the caller is different.

the protocol adapters are responsible for:
- exposing an endpoint that can be called
- parsing the request from the caller
- authn and authz of the caller
- populating context variables with auth context
- calling the appropriate agent run formatter
- parsing the response from the formatter and returning it to the caller

the protocol adapters never directly call the agent - they always call the formatter (which in turn runs the agent).
"""