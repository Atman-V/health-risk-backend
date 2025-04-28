const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const authRoute = require("./routes/authRoute");

const app = express();
app.use(cors());
app.use(express.json());

mongoose.connect("mongodb+srv://sabariS:Q5dQQb31dzzwZ2mL@cluster0.tdgspkd.mongodb.net/healthrisk?retryWrites=true&w=majority", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
}).then(() => console.log("MongoDB connected"))
  .catch((err) => console.log("DB error:", err));

// Routes
app.use("/auth", authRoute);

// Start server
app.listen(9000, () => console.log("Auth Server running on http://localhost:9000"));
