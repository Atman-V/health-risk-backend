const express = require("express");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const User = require("../models/user");

const router = express.Router();

// REGISTER
router.post("/register", async (req, res) => {
  try {
    const { username, password } = req.body;
    console.log("📩 Register request:", username);

    if (!username || !password) {
      return res.status(400).json({ message: "Username and password required" });
    }

    const userExists = await User.findOne({ username });
    if (userExists) {
      console.log("⚠️ Username already exists");
      return res.status(400).json({ message: "User already exists" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ username, password: hashedPassword });

    await newUser.save();
    console.log("✅ Registered user:", username);

    return res.status(201).json({ message: "Registered successfully" });

  } catch (err) {
    console.error("❌ Register Error:", err);
    return res.status(500).json({ message: "Internal server error", error: err.message });
  }
});

// LOGIN
router.post("/login", async (req, res) => {
  try {
    const { username, password } = req.body;
    console.log("🔐 Login request:", username);

    if (!username || !password) {
      return res.status(400).json({ message: "Username and password required" });
    }

    const user = await User.findOne({ username });
    if (!user) {
      console.log("❌ User not found");
      return res.status(404).json({ message: "User not found" });
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      console.log("❌ Invalid password");
      return res.status(401).json({ message: "Invalid password" });
    }

    // Hardcoded JWT secret key
    const token = jwt.sign({ username }, "sabaris_secret", { expiresIn: "1h" });
    console.log("✅ Login success");

    return res.status(200).json({ message: "Login successful", token });

  } catch (err) {
    console.error("❌ Login Error:", err);
    return res.status(500).json({ message: "Internal server error", error: err.message });
  }
});

module.exports = router;
