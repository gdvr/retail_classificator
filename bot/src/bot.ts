import axios from "axios";
import "dotenv/config";
import { Telegraf } from "telegraf";

const bot = new Telegraf(process.env.TOKEN as string);
const CHAT_ID = "1520311400";

interface MessageResponse {
  messages: string[];
}

const sendToTelegram = async (
  messages: string[],
  chatId: string,
  bot: Telegraf
): Promise<void> => {
  const maxLength = 4096; // Telegram max message length

  // Group messages in sets of 3
  const groupedMessages = [];
  for (let i = 0; i < messages.length; i += 3) {
    groupedMessages.push(messages.slice(i, i + 3).join("\n\n"));
  }

  for (const message of groupedMessages) {
    // Split the message into chunks if it exceeds the max length
    const chunks = message.match(new RegExp(`.{1,${maxLength}}`, "g")) || [];

    for (const chunk of chunks) {
      try {
        // Send each chunk
        await bot.telegram.sendMessage(chatId, chunk);
        console.log(`Message sent: ${chunk}`);

        // Wait 2 seconds before sending the next message
        await new Promise((resolve) => setTimeout(resolve, 2000));
      } catch (error) {
        console.error(`Failed to send message: ${chunk}`, error);
      }
    }
  }
};

// Function to fetch data from the API and process the messages
const fetchAndNotify = async (
  fecha: string,
  storeId: string,
  chatId: string,
  bot: Telegraf
): Promise<void> => {
  try {
    // Make a GET request to the API endpoint
    const response = await axios.get<MessageResponse>(
      `http://localhost:5000/notify`,
      {
        params: {
          fecha: fecha,
          store_id: storeId,
        },
      }
    );

    const messages = response.data.messages;

    // Send the messages to Telegram
    if (messages.length > 0) {
      await sendToTelegram(messages, chatId, bot);
    } else {
      console.log("No messages to send.");
    }
  } catch (error) {
    console.error("Error fetching data from the API:", error);
  }
};

bot.start((ctx) => ctx.reply("Welcome! 👋 I am your bot."));
bot.help((ctx) => ctx.reply("Send me any message, and I will echo it back!"));

// Error handling
bot.catch((err) => {
  console.error("An error occurred:", err);
});

const fecha = "2023-06-30T00:00:00"; // Example fecha parameter
const storeId = "KQZZGB09"; // Example store_id parameter

fetchAndNotify(fecha, storeId, CHAT_ID, bot);

// Start the bot (optional, depending on if you need it to listen for messages)
bot.launch().then(() => console.log("Bot is running..."));
// Graceful stop
process.once("SIGINT", () => bot.stop("SIGINT"));
process.once("SIGTERM", () => bot.stop("SIGTERM"));
