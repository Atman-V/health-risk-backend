const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const authRoute = require("./routes/authRoute");

const app = express();

// Middleware
app.use(cors()); // Enable CORS
app.use(express.json()); // Parse JSON bodies

// MongoDB connection using the hardcoded URI
const dbURI = "mongodb+srv://sabariS:Q5dQQb31dzzwZ2mL@cluster0.tdgspkd.mongodb.net/healthrisk?retryWrites=true&w=majority";

mongoose.connect(dbURI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
}).then(() => {
  console.log("✅ MongoDB connected");
}).catch((err) => {
  console.log("❌ DB error:", err);
});

// Routes
app.use("/auth", authRoute); // Use auth routes for authentication

// Start the server using the PORT environment variable, default to 9000
const port = process.env.PORT || 9000;
app.listen(port, () => {
  console.log(`Auth Server running on http://localhost:${port}`);
});
