"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const telegraf_1 = require("telegraf");
// Replace 'YOUR_HTTP_API' with your actual Telegram Bot HTTP API Token
const bot = new telegraf_1.Telegraf("7875851710:AAHL7H6mSkGyDkZWA-QXJRBcZl16Ic_w4Og");
// Basic command handlers
bot.start((ctx) => ctx.reply("Welcome! 👋 I am your bot."));
bot.help((ctx) => ctx.reply("Send me any message, and I will echo it back!"));
// Echo all messages
bot.on("text", (ctx) => {
    ctx.reply(`You said: ${ctx.message.text}`);
});
// Error handling
bot.catch((err) => {
    console.error("An error occurred:", err);
});
// Start the bot
bot.launch().then(() => {
    console.log("Bot is running...");
});
// Graceful stop
process.once("SIGINT", () => bot.stop("SIGINT"));
process.once("SIGTERM", () => bot.stop("SIGTERM"));
