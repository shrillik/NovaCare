from graphviz import Digraph

dot = Digraph("AI_First_Customer_Support_HLD", format="png")
dot.attr(rankdir="TB", size="10,8")

# Define nodes (system components)
dot.node("U", "User\n(Text/Voice)", shape="oval", style="filled", color="#87CEFA")
dot.node("F", "Frontend (React.js)\nChat + Voice UI", shape="box", style="filled", color="#90EE90")
dot.node("A", "Backend API Gateway (Flask)\nRequest Routing & Processing", shape="box", style="filled", color="#FFD700")
dot.node("N", "NLP Processor\nText Cleaning + Tokenization", shape="box", style="filled", color="#FFB6C1")
dot.node("M", "ML Engine\n(Intent Classification + Response Generation + Sentiment Detection)", shape="box", style="filled", color="#FF7F50")
dot.node("C", "Context Manager\nMaintains Conversation Memory", shape="box", style="filled", color="#DDA0DD")
dot.node("E", "Escalation Handler\n(Human Support)", shape="box", style="filled", color="#FFA07A")
dot.node("R", "Response to User\n(Text/Voice Output)", shape="oval", style="filled", color="#87CEFA")

# Voice processing
dot.node("S1", "SpeechRecognition\n(Voice → Text)", shape="ellipse", style="filled", color="#E6E6FA")
dot.node("S2", "pyttsx3\n(Text → Speech)", shape="ellipse", style="filled", color="#E6E6FA")

# Define edges
dot.edge("U", "S1", label="Voice Input")
dot.edge("U", "F", label="Text Input")
dot.edge("S1", "F", label="Converted Text")

dot.edge("F", "A", label="User Query")
dot.edge("A", "N", label="Preprocess")
dot.edge("N", "M", label="Classify Intent / Generate Response")
dot.edge("M", "C", label="Update Context")

dot.edge("M", "E", label="Low Confidence", color="red")
dot.edge("M", "R", label="Response Generated")
dot.edge("E", "R", label="Human Response")
dot.edge("R", "S2", label="Convert to Speech")
dot.edge("R", "U", label="Text Output")
dot.edge("S2", "U", label="Voice Output")

# Render diagram
dot.render("AI_First_Customer_Support_HLD", view=True)
