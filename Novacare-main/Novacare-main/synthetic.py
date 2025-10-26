import random
import pandas as pd

# Define intents and templates
intents = {
    "check_bill": [
        "I want to check my bill",
        "Show me my last bill",
        "How much do I owe this month?"
    ],
    "billing_complaint": [
        "Why is my bill so high?",
        "I was overcharged",
        "Incorrect bill amount"
    ],
    "recharge": [
        "Recharge my account with 200 INR",
        "Top up 50 INR",
        "Add balance to my number"
    ],
    "check_balance": [
        "Check my remaining balance",
        "How much balance do I have?",
        "Account balance"
    ],
    "plan_change": [
        "Upgrade to 5GB plan",
        "Switch my plan",
        "Change my data plan"
    ],
    "plan_inquiry": [
        "What plans are available?",
        "Show me your plans",
        "Data plans info"
    ],
    "network_issue": [
        "Internet not working",
        "No signal at home",
        "My network is down"
    ],
    "account_help": [
        "How do I change my registered number?",
        "Reset my account password",
        "Update account info"
    ],
    "cancel_subscription": [
        "Cancel my subscription",
        "Stop my service",
        "I want to unsubscribe"
    ],
    "roaming_help": [
        "Activate roaming",
        "How do I use my phone abroad?",
        "Roaming charges info"
    ],
    "plan_inquiry": [
        "What plans are available?",
        "Show me your plans",
        "Data plans info",
"Show me recharge plans for 500 rupees",
"I want 1.5GB per day for 30 days",
"Which plan should I choose for 1000 INR?",
"Give me 10GB data plan",
"Internet plan valid for 7 days"
    ]
}

# Generate synthetic dataset
data = []
for intent, templates in intents.items():
    for _ in range(50):  # 50 examples per intent
        query = random.choice(templates)
        # Small variation: add “please” or “help me” sometimes
        if random.random() > 0.7:
            query = query + " please"
        if random.random() > 0.8:
            query = "Can you " + query.lower() + "?"
        data.append({"query": query, "intent": intent})

# Shuffle dataset
random.shuffle(data)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("synthetic_telecom_dataset.csv", index=False)
print("Synthetic telecom dataset created: synthetic_telecom_dataset.csv")
