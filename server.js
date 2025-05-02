const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const authRoute = require("./routes/authRoute"); // Import auth routes

const app = express();

// Middleware
app.use(cors()); // Enable CORS for cross-origin requests
app.use(express.json()); // Parse JSON bodies

// MongoDB connection (hardcoded URI without deprecated options)
const dbURI = "mongodb+srv://sabariS:Q5dQQb31dzzwZ2mL@cluster0.tdgspkd.mongodb.net/healthrisk?retryWrites=true&w=majority";

mongoose
  .connect(dbURI) // Removed deprecated options
  .then(() => {
    console.log("✅ MongoDB connected");
  })
  .catch((err) => {
    console.log("❌ DB connection error:", err);
  });

// Routes
app.use("/auth", authRoute); // Use auth routes for authentication

// Start server (hardcoded PORT)
const port = 9000;
app.listen(port, () => {
  console.log(`Auth Server running on http://localhost:${port}`);
});
