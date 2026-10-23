"""
Discord Expanded Privacy Taxonomy
==================================
Sources:
1. Akgul et al. (USENIX Security 2024) - 20 themes inherited from prior work
2. Our inductive coding - 11 new themes (2 Security & Safety, 9 Discord-specific)

Total: 31 themes, ~139 fine-grained aspects

This module is the single source of truth for the taxonomy. Theme names and
category assignments here match Table V and Figure 4 of the paper exactly.
"""

PRIVACY_TAXONOMY = {

    # =========================================================================
    # AKGUL CATEGORY 1: Data Practices (7 concepts)
    # =========================================================================

    "Data Collection": {
        "category": "Data Practices",
        "source": "Akgul",
        "description": "What information is collected from users",
        "aspects": [
            "Collecting unnecessary personal data",
            "Purpose of data collection",
            "Data aggregation",
            "Data minimization",
        ]
    },
    "Data Sharing": {
        "category": "Data Practices",
        "source": "Akgul",
        "description": "How data is shared with others",
        "aspects": [
            "Accidental sharing",
            "Forced sharing",
            "Unintended sharing",
            "Cross-app sharing",
            "Secondary use",
            "Calendar sharing",
            "Password protected sharing",
        ]
    },
    "Data Deletion": {
        "category": "Data Practices",
        "source": "Akgul",
        "description": "Ability to delete personal data",
        "aspects": [
            "Browsing history removal",
            "Delete video history",
            "Delete search history",
            "Cookie removal",
            "Photo deletion",
            "Video deletion",
            "File deletion",
            "Profile deletion",
        ]
    },
    "Remove Personally Identifiable Information": {
        "category": "Data Practices",
        "source": "Akgul",
        "description": "Removing PII from stored data",
        "aspects": [
            "Remove name",
            "Remove address",
            "Remove phone number",
            "Remove personal information",
            "Remove photo",
        ]
    },
    "Data Exposure": {
        "category": "Data Practices",
        "source": "Akgul",
        "description": "Data exposed to unintended parties",
        "aspects": [
            "To advertisers",
            "To app developers",
            "Public accessibility",
            "Data disclosure",
            "Safety",
            "Use limitation",
        ]
    },
    "Data Hiding": {
        "category": "Data Practices",
        "source": "Akgul",
        "description": "Ability to hide personal data from others",
        "aspects": [
            "Name",
            "Video",
            "Events",
            "Albums",
            "Photos",
            "Contacts",
            "Display name",
            "Email",
            "Messages",
            "Notes",
            "Playlist",
            "Folders",
            "Meetings",
        ]
    },
    "Location and Tracking": {
        "category": "Data Practices",
        "source": "Akgul",
        "description": "Location data collection and user tracking",
        "aspects": [
            "Location data",
            "Tracking",
        ]
    },

    # =========================================================================
    # AKGUL CATEGORY 2: User Rights & Controls (7 concepts)
    # =========================================================================

    "Consent": {
        "category": "User Rights & Controls",
        "source": "Akgul",
        "description": "Whether the user agreed to data practices",
        "aspects": [
            "Giving consent",
            "Agreement",
            "Authorize",
            "Consent process",
            "Opt-out (notice awareness)",
            "Without consent",
            "Mis-activation",
            "Forced consent",
        ]
    },
    "Privacy Controls": {
        "category": "User Rights & Controls",
        "source": "Akgul",
        "description": "User-facing privacy settings and controls",
        "aspects": [
            "Privacy settings",
            "Changing personal information",
            "Location sharing",
            "Download my data",
            "Parental controls",
            "Privacy defaults",
            "Password protected controls",
        ]
    },
    "Anonymity / Identification": {
        "category": "User Rights & Controls",
        "source": "Akgul",
        "description": "User anonymity and identification concerns",
        "aspects": [
            "User wants to be anonymous",
            "Incognito",
            "Hide group participants",
            "Fear of identification",
            "Misattribution",
        ]
    },
    "Advertising": {
        "category": "User Rights & Controls",
        "source": "Akgul",
        "description": "Ad-related privacy concerns",
        "aspects": [
            "Ads personalization",
            "Paid services",
        ]
    },
    "Privacy Policies and Laws": {
        "category": "User Rights & Controls",
        "source": "Akgul",
        "description": "Compliance with privacy laws and policies",
        "aspects": [
            "Privacy policy concerns",
            "Legal compliance",
        ]
    },
    "Privacy Invasion": {
        "category": "User Rights & Controls",
        "source": "Akgul",
        "description": "General privacy invasion concerns",
        "aspects": [
            "General privacy invasion",
        ]
    },
    "Positive Privacy": {
        "category": "User Rights & Controls",
        "source": "Akgul",
        "description": "Users praising privacy features or transparent behavior",
        "aspects": [
            "Positive privacy feedback",
        ]
    },

    # =========================================================================
    # AKGUL CATEGORY 3: Security & Safety (8 concepts)
    # =========================================================================

    "Data Security": {
        "category": "Security & Safety",
        "source": "Akgul",
        "description": "Security of stored and transmitted data",
        "aspects": [
            "Safety",
            "Protection",
            "Breach of confidentiality",
            "Data breach",
            "Account hacking",
            "Fake profiles",
        ]
    },
    "Password Issues": {
        "category": "Security & Safety",
        "source": "Akgul",
        "description": "Password and credential management concerns",
        "aspects": [
            "Password management concerns",
        ]
    },
    "Data Accuracy": {
        "category": "Security & Safety",
        "source": "Akgul",
        "description": "Accuracy and correctness of stored data",
        "aspects": [
            "Inaccurate",
            "Obsolete",
            "Wrong association",
        ]
    },
    "Safety": {
        "category": "Security & Safety",
        "source": "Akgul",
        "description": "User safety threats from data misuse",
        "aspects": [
            "Blackmail",
            "Appropriation",
            "Intrusion",
            "Decisional interference",
        ]
    },
    "Selling Data": {
        "category": "Security & Safety",
        "source": "Akgul",
        "description": "Selling user data for profit",
        "aspects": [
            "To 3rd parties",
            "For ads purposes",
        ]
    },
    "Surveillance": {
        "category": "Security & Safety",
        "source": "Akgul",
        "description": "Monitoring and surveillance of users",
        "aspects": [
            "Spying",
            "Stalkerware",
            "Satellite",
        ]
    },

    # =========================================================================
    # CATEGORY 4: Discord-Specific Risks (11 concepts) — NEW
    # =========================================================================

    "Over-Listening": {
        "category": "Discord-Specific Risks",
        "source": "New",
        "description": "Passive monitoring of messages without being invoked",
        "aspects": [
            "Reading messages in unrelated channels",
            "Processing messages without being called",
            "Background monitoring of all messages",
            "Listening beyond stated scope",
            "Monitoring activity when not invoked",
        ]
    },
    "Permission Scope Mismatch": {
        "category": "Discord-Specific Risks",
        "source": "New",
        "description": "Requesting high-level permissions for low-level tasks",
        "aspects": [
            "Requesting admin permissions unnecessarily",
            "Over-broad permission scopes",
            "Permissions don't match bot functionality",
            "Requesting dangerous permissions (ban, kick) for utility bot",
            "Music bot requesting message read access",
        ]
    },
    "Message Content Access": {
        "category": "Discord-Specific Risks",
        "source": "New",
        "description": "Use of the privileged MESSAGE_CONTENT intent",
        "aspects": [
            "Bot reads all messages via MESSAGE_CONTENT intent",
            "Message content intent used unnecessarily",
            "Bot can see private conversations in channels",
            "Logging message content without disclosure",
        ]
    },
    "DM Privacy": {
        "category": "Discord-Specific Risks",
        "source": "New",
        "description": "Bots messaging users privately without consent or logging DMs",
        "aspects": [
            "Unsolicited DMs from bot",
            "Bot logging DM conversations",
            "Spam DMs after adding bot",
            "Bot reads DMs without need",
            "DM advertising/promotion",
        ]
    },
    "Data Retention": {
        "category": "Security & Safety",
        "source": "New",
        "description": "Keeping logs of messages, users, or activity indefinitely",
        "aspects": [
            "Message logs kept indefinitely",
            "User data not deleted after leaving server",
            "Data persists after bot is kicked",
            "No data retention policy",
            "Chat history stored permanently",
        ]
    },
    "Gateway Intent Abuse": {
        "category": "Discord-Specific Risks",
        "source": "New",
        "description": "Misusing Discord gateway intents like GUILD_PRESENCE or GUILD_MEMBERS",
        "aspects": [
            "Tracking online/offline status via GUILD_PRESENCE",
            "Accessing full member list via GUILD_MEMBERS",
            "Using privileged intents beyond stated purpose",
            "Monitoring user presence patterns",
        ]
    },
    "Webhook Misuse": {
        "category": "Discord-Specific Risks",
        "source": "New",
        "description": "Creating or using webhooks to bypass audit logs or impersonate",
        "aspects": [
            "Creating webhooks without permission",
            "Impersonating users via webhooks",
            "Bypassing server audit logs",
            "Webhook spam",
        ]
    },
    "Server Profiling": {
        "category": "Discord-Specific Risks",
        "source": "New",
        "description": "Mapping the structure, membership, roles, and activity of a server",
        "aspects": [
            "Harvesting server member lists",
            "Mapping server structure and roles",
            "Collecting server analytics without consent",
            "Exporting server data",
        ]
    },
    "Voice Privacy": {
        "category": "Discord-Specific Risks",
        "source": "New",
        "description": "Unauthorized recording or monitoring of voice channels",
        "aspects": [
            "Voice recording without consent",
            "Voice data stored without notice",
            "Music bot recording conversations",
            "Voice activity monitoring",
            "Voice data shared with third parties",
        ]
    },
    "Phishing and Scams": {
        "category": "Security & Safety",
        "source": "New",
        "description": "Fake links to steal accounts or tokens",
        "aspects": [
            "Fake Nitro gift links",
            "Fake verification links",
            "Token grabbing attempts",
            "Credential harvesting",
            "Scam links in bot messages",
        ]
    },
    "Bot Moderation Transparency": {
        "category": "Discord-Specific Risks",
        "source": "New",
        "description": "Clarity and fairness of automated moderation actions",
        "aspects": [
            "No reason given for ban/kick",
            "Automated moderation too aggressive",
            "False positive moderation actions",
            "No appeal process for bot actions",
            "Moderation logs not accessible to users",
        ]
    },
}

# =============================================================================
# Category groupings
# =============================================================================

CATEGORIES = {
    "Data Practices": [
        "Data Collection", "Data Sharing", "Data Deletion",
        "Remove Personally Identifiable Information", "Data Exposure",
        "Data Hiding", "Location and Tracking",
    ],
    "User Rights & Controls": [
        "Consent", "Privacy Controls", "Anonymity / Identification",
        "Advertising", "Privacy Policies and Laws", "Privacy Invasion",
        "Positive Privacy",
    ],
    "Security & Safety": [
        "Data Security", "Password Issues", "Data Accuracy",
        "Safety", "Selling Data", "Surveillance",
        "Phishing and Scams", "Data Retention",
    ],
    "Discord-Specific Risks": [
        "Over-Listening", "Permission Scope Mismatch", "Message Content Access",
        "DM Privacy", "Gateway Intent Abuse", "Webhook Misuse",
        "Server Profiling", "Voice Privacy", "Bot Moderation Transparency",
    ],
}

# Flat list of all theme names
ALL_THEMES = list(PRIVACY_TAXONOMY.keys())

# Flat list of all fine-grained aspects
ALL_ASPECTS = []
for theme, details in PRIVACY_TAXONOMY.items():
    for aspect in details["aspects"]:
        ALL_ASPECTS.append(f"{theme}: {aspect}")


def get_taxonomy_prompt_text():
    """Generate a text description of the taxonomy for LLM prompts."""
    lines = []
    current_category = None
    for theme, details in PRIVACY_TAXONOMY.items():
        if details["category"] != current_category:
            current_category = details["category"]
            lines.append(f"\n=== {current_category} ===")
        lines.append(f"\n## {theme}")
        lines.append(f"Description: {details['description']}")
        lines.append("Fine-grained aspects:")
        for aspect in details["aspects"]:
            lines.append(f"  - {aspect}")
    return "\n".join(lines)


def get_theme_list_prompt():
    """Generate a compact theme list for classification prompts."""
    lines = []
    for theme, details in PRIVACY_TAXONOMY.items():
        lines.append(f"- {theme}: {details['description']}")
    return "\n".join(lines)


# =============================================================================
# Emotion groups (from Ekman's taxonomy, matching Akgul et al.)
# =============================================================================

EMOTION_GROUPS = {
    "Anger": ["anger", "annoyance"],
    "Disgust": ["disgust"],
    "Fear": ["fear", "nervousness"],
    "Joy": ["joy", "amusement", "excitement", "love", "gratitude",
            "relief", "pride", "admiration", "optimism"],
    "Sadness": ["sadness", "grief", "disappointment", "remorse"],
    "Surprise": ["surprise", "realization", "curiosity"],
    "Confusion": ["confusion", "embarrassment"],
    "Neutral": ["neutral", "approval", "disapproval", "caring", "desire"],
}