import { Telegraf } from "telegraf";

// Replace 'YOUR_HTTP_API' with your actual Telegram Bot HTTP API Token
const bot = new Telegraf("7875851710:AAHL7H6mSkGyDkZWA-QXJRBcZl16Ic_w4Og");

// Basic command handlers
bot.start((ctx) => ctx.reply("Welcome! 👋 I am your bot."));
bot.help((ctx) => ctx.reply("Send me any message, and I will echo it back!"));

// Echo all messages
/*bot.on("text", (ctx) => {
  ctx.reply(`You said: ${ctx.message.text}`);
});*/

// Error handling
bot.catch((err) => {
  console.error("An error occurred:", err);
});

bot.hears("menu", (ctx) => {
  ctx.reply("Choose an option:", {
    reply_markup: {
      inline_keyboard: [
        [{ text: "Option 1", callback_data: "option1" }],
        [{ text: "Option 2", callback_data: "option2" }],
      ],
    },
  });
});

bot.on("callback_query", (ctx) => {
  const data = ctx.callbackQuery?.message;
  console.log((ctx.callbackQuery as any)?.data);
  ctx.reply(`You selected: ${data}`);
});

// Start the bot
bot.launch().then(() => {
  console.log("Bot is running...");
});

// Graceful stop
process.once("SIGINT", () => bot.stop("SIGINT"));
process.once("SIGTERM", () => bot.stop("SIGTERM"));
