from playwright.async_api import async_playwright
import asyncio 
# import time 
import random
import gspread
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import pandas as pd 
import csv 
from temporalio import activity


load_dotenv()
def csv_append(file_path, row):
    with open(file_path, 'a') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(row)
        f_object.close()

async def copy_image(context, img_uri):
    page_3 = await context.new_page()
    await page_3.goto(img_uri)
    await asyncio.sleep(1)
    await page_3.keyboard.press("Control+C")
    await asyncio.sleep(1)
    try: 
        clipboard = await page_3.evaluate("navigator.clipboard.readText()")
    except:
        clipboard = None
    await page_3.close()
    return clipboard

async def get_img_text(page_1, context, img_url):

    await copy_image(context, img_url)
    await page_1.goto('https://ifimageediting.com/image-to-text')
    await asyncio.sleep(5)         
    await page_1.keyboard.down("PageDown")
    await asyncio.sleep(2) 
      
    box = await page_1.query_selector('img.img-fluid[alt="Image To Text"]')
    await asyncio.sleep(2)   
    print(box)
    if box:
        await box.click()
    # print("Click")
    await asyncio.sleep(2)   
    await page_1.keyboard.press("Control+V")
    # print("Control V")
    await page_1.keyboard.down("PageDown")
    await asyncio.sleep(25)

    button = await page_1.query_selector("[id='iie_submission']")
    await asyncio.sleep(2)
    await button.click()
    await asyncio.sleep(random.randint(50, 60))
    await page_1.keyboard.down("PageDown")
    await asyncio.sleep(2)           
    for _ in range(3):
        try:   
            text = await page_1.query_selector(".text_box.text-start > p:nth-child(1) > xmp:nth-child(1)")
            await asyncio.sleep(2)
            text = await text.inner_text()
            await asyncio.sleep(2)
            return text 
        except Exception as e:
            print(e)
            await asyncio.sleep(5)
    return None 

async def crawl_new_chapter(cookie_path, link_gr, limit_posts, start_idx ):
    with open(cookie_path, 'r') as f:
        cookie = json.load(f)
    df = pd.read_csv('raw_data.csv')
    column = df['url'].tolist()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch( 
                                                headless=False,
                                                )
        # userAgent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        userAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        context = await browser.new_context(
            # proxy={
            #     "server": "http://192.168.0.108:3128",
            # },
            user_agent=userAgent,
            permissions=["clipboard-read", "clipboard-write"],
            no_viewport=True)
 
        await context.add_cookies(cookie)
        await asyncio.sleep(2)
        page = await context.new_page()
        page.set_default_timeout(0)
        await page.goto("https://www.facebook.com")
        await asyncio.sleep(2)
        await page.goto(link_gr)
        await page.keyboard.down("PageDown")
        await asyncio.sleep(2)
        post_index = start_idx
        for _ in range(start_idx):
            await page.keyboard.down("PageDown")
            await asyncio.sleep(5)
        cookie = await page.context.cookies()
        with open(cookie_path, "w") as outfile:
            outfile.write(json.dumps(cookie))
        await asyncio.sleep(2)
        feed = await page.query_selector('.x6s0dn4.x78zum5.xdt5ytf.x193iq5w > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > div:nth-child(3)')
        await asyncio.sleep(2)
        maxtry = 3
        trytime = 0 
        while post_index <= limit_posts and trytime < maxtry:
            for _ in range(3):
                try:
                    post = await feed.query_selector(' > div:nth-child({})'.format(post_index))
                    await asyncio.sleep(2)
                except:
                    feed = await page.query_selector('.x6s0dn4.x78zum5.xdt5ytf.x193iq5w > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > div:nth-child(3)')
                    await asyncio.sleep(2)
                if post:
                    break 
            await asyncio.sleep(2)
            if post is None:
                post_index += 1
                await page.keyboard.down("PageDown")
                await asyncio.sleep(5)
                trytime += 1
                continue 
            trytime = 0
            for _ in range(3):
                try:
                    link = await post.query_selector('.x1i10hfl.xjbqb8w.x1ejq31n.xd10rxx.x1sy0etr.x17r0tee.x972fbf.xcfux6l.x1qhh985.xm0m39n.x9f619.x1ypdohk.xt0psk2.xe8uvvx.xdj266r.x11i5rnm.xat24cr.x1mh8g0r.xexx8yu.x4uap5.x18d9i69.xkhd6sd.x16tdsg8.x1hl2dhg.xggy1nq.x1a2a7pz.x1heor9g.xt0b8zv.xo1l8bm')
                    await asyncio.sleep(2)
                except:
                    pass
                if link:
                    break 
            if link == None:
                await page.keyboard.down("PageDown")
                await asyncio.sleep(5)
                post_index += 1
                continue   
                                
            content = await post.query_selector('.x1iorvi4.x1pi30zi.x1l90r2v.x1swvt13 > div:nth-child(1) > div:nth-child(1) > span:nth-child(1)')
            await link.click(button='right')
            await asyncio.sleep(2)
            href = await link.get_attribute('href')
            await asyncio.sleep(2)
            if href == '#' or href.split('?')[0] in column:
                post_index += 1
                continue 
            href = href.split('?')[0]
            
            if content:
                print("Content")
                text = []
                for _ in range(3):
                    try:
                        more = await content.query_selector('.x1i10hfl.xjbqb8w.x1ejq31n.xd10rxx.x1sy0etr.x17r0tee.x972fbf.xcfux6l.x1qhh985.xm0m39n.x9f619.x1ypdohk.xt0psk2.xe8uvvx.xdj266r.x11i5rnm.xat24cr.x1mh8g0r.xexx8yu.x4uap5.x18d9i69.xkhd6sd.x16tdsg8.x1hl2dhg.xggy1nq.x1a2a7pz.xt0b8zv.xzsf02u.x1s688f')
                        await asyncio.sleep(2)
                    except:
                        await page.keyboard.down("PageDown")
                        await asyncio.sleep(5)
                    if more:
                        await more.click()
                        await asyncio.sleep(2)
                        break 
                
                content = await content.inner_text()   
                await asyncio.sleep(2) 
                # print(href, content)
                if len(content) < 100:
                    post_index += 1
                    continue 
                csv_append("raw_data.csv", [href, content, 1])
                
            img = await post.query_selector('img[alt="Có thể là hình ảnh về văn bản"]')
            await asyncio.sleep(2)
            if img:
                print("Image")
                print(href, img)
                await asyncio.sleep(5) 
                if img:
                    page_1 = await context.new_page()
                    img_url = await img.get_attribute('src')
                    await asyncio.sleep(2) 
                    print(img_url)
                    if img_url:
                        content = await get_img_text(page_1, context, img_url)
                    await page_1.close()
                if content:
                    print(href, content)
                    if len(content) < 200:
                        post_index += 1
                        continue 
                    # csv_append("fb_posts.csv", [href, content, 1])
                else:
                    print("Can not extract image to text")
                
            await page.keyboard.down("PageDown")
            await asyncio.sleep(3)
            post_index += 1
            
        await browser.close()


async def main_activity():
    link_gr = 'https://www.facebook.com/nhanvangiaiphamvn'
    await crawl_new_chapter('cookie3.json', link_gr, 100, 1)

if __name__ == "__main__":
    asyncio.run(main_activity())