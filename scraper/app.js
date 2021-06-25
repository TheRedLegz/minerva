const pp = require("puppeteer");
const fs = require("fs");

// goto the twitter search url - done
// get the adaptive json response - done
// save it - done
// check scroll position - if cant scroll, quit. if can, continue /
// scroll /
// check for adaptive - if yes, go back to step 2. if no, go back to step 4 /

// save as json for checking
// TODO connect to db

const db = [];

const scrape = async (keywords = null, dateFrom = null, dateTo = null) => {
  const browser = await pp.launch();
  const page = await browser.newPage();

  page.on("response", async (e) => {
    const url = e.url();

    if (url.includes("adaptive.json")) {
      const json = await e.json();

      const keys = Object.keys(json.globalObjects.tweets);

      for (const i of keys) {
        db.push(json.globalObjects.tweets[i]);
      }
    }
  });

  await page.setUserAgent(
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36 Edg/91.0.864.54"
  );

  await page.goto(
    "https://twitter.com/search?q=(from%3Aelonmusk)%20until%3A2021-06-23%20since%3A2021-06-01%20-filter%3Areplies",
    {
      waitUntil: "networkidle0",
    }
  );

  let lastHeight = await page.evaluate(() => {
    return document.body.scrollHeight;
  });

  while (true) {
    try {
      await page.evaluate((y) => {
        window.scrollTo(0, y);
      }, lastHeight);

      await page.waitForTimeout(2000);

      const currentHeight = await page.evaluate(() => {
        return document.body.scrollHeight;
      });

      if (lastHeight == currentHeight) break;

      lastHeight = currentHeight;
    } catch (e) {
      console.log(e);
      break;
    }
  }

  fs.writeFileSync('sample.json', JSON.stringify(db, null, 2))

  await browser.close();
};

scrape();
