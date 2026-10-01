"""Bundled lexicons the curation stages read: stopword lists and adult/spam term lists.

Their bytes fold into the sentinel hash of the stage that loads them, so editing a
list reruns that stage.
"""
