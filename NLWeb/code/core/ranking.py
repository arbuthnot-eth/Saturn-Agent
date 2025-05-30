# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
This file contains the code for the ranking stage. 

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

from utils.utils import log
from llm.llm import ask_llm
import asyncio
import json
from utils.trim import trim_json
from prompts.prompts import find_prompt, fill_ranking_prompt
from utils.logging_config_helper import get_configured_logger

logger = get_configured_logger("ranking_engine")


class Ranking:
     
    SCORE_THRESHOLD = 60 # New threshold for relevance
    NUM_RESULTS_TO_SEND = 10

    FAST_TRACK = 1
    REGULAR_TRACK = 2

    # This is the default ranking prompt, in case, for some reason, we can't find the site_type.xml file.
    RANKING_PROMPT_TEMPLATE = ["""You are a relevance scoring expert. Assign a relevance score from 0 to 100 to the given {site.itemType} based *strictly* on how directly and comprehensively it answers the user's question: "{request.query}".
The item's own description is: {item.description}.

- If the score is {{SCORE_THRESHOLD}} or above, the item is considered relevant. Provide a concise yet informative description (1-2 sentences) explaining *why* it is relevant to the user's question, without mentioning the user's question itself or the score. Aim to capture the essence of the item's contribution to answering the query.
- If the score is below {{SCORE_THRESHOLD}}, the item is considered not relevant. For the description, simply state: "This item does not directly answer the user's query." Do not attempt to justify its relevance.

Output only the score and the description as per the structure below.
""",
    {"score" : "integer between 0 and 100", 
 "description" : "concise and informative description based on relevance score, or a fixed string if not relevant"}]
 
    RANKING_PROMPT_NAME = "RankingPrompt"
     
    def get_ranking_prompt(self):
        site = self.handler.site
        item_type = self.handler.item_type
        # find_prompt returns the template string and ans_structure from site_type.xml if found
        prompt_template_from_xml, ans_struc_from_xml = find_prompt(site, item_type, self.RANKING_PROMPT_NAME)

        if prompt_template_from_xml is None:
            logger.debug(f"Using default ranking prompt for site: {site}, item_type: {item_type}")
            # Format the default template with the current SCORE_THRESHOLD
            prompt_str = self.RANKING_PROMPT_TEMPLATE[0].replace("{{SCORE_THRESHOLD}}", str(self.SCORE_THRESHOLD))
            ans_struc = self.RANKING_PROMPT_TEMPLATE[1]
        else:
            logger.debug(f"Using custom ranking prompt from XML for site: {site}, item_type: {item_type}")
            # If custom prompt from XML contains {{SCORE_THRESHOLD}}, format it. Otherwise, use as is.
            if "{{SCORE_THRESHOLD}}" in prompt_template_from_xml:
                 prompt_str = prompt_template_from_xml.replace("{{SCORE_THRESHOLD}}", str(self.SCORE_THRESHOLD))
            else:
                prompt_str = prompt_template_from_xml
            ans_struc = ans_struc_from_xml
        return prompt_str, ans_struc
        
    def __init__(self, handler, items, ranking_type=FAST_TRACK):
        ll = len(items)
        self.ranking_type_str = "FAST_TRACK" if ranking_type == self.FAST_TRACK else "REGULAR_TRACK"
        logger.info(f"Initializing Ranking with {ll} items, type: {self.ranking_type_str}")
        logger.info(f"Ranking {ll} items of type {self.ranking_type_str}")
        self.handler = handler
        self.items = items
        self.num_results_sent = 0
        self.rankedAnswers = []
        self.ranking_type = ranking_type
        self._results_lock = asyncio.Lock()  # Add lock for thread-safe operations
        self.early_sent_titles = set() # To track titles of early-sent items

    async def rankItem(self, url, json_str, name, site):
        if not self.handler.connection_alive_event.is_set():
            logger.warning("Connection lost, skipping item ranking")
            return
        if (self.ranking_type == Ranking.FAST_TRACK and self.handler.abort_fast_track_event.is_set()):
            logger.info("Fast track aborted, skipping item ranking")
            logger.info("Aborting fast track")
            return
        try:
            logger.debug(f"Ranking item: {name} from {site}")
            prompt_str, ans_struc = self.get_ranking_prompt()
            description = trim_json(json_str)
            prompt = fill_ranking_prompt(prompt_str, self.handler, description)
            
            logger.debug(f"Sending ranking request to LLM for item: {name}")
            ranking = await ask_llm(prompt, ans_struc, level="low")
            logger.debug(f"Received ranking score: {ranking.get('score', 'N/A')} for item: {name}")
            
            ansr = {
                'url': url,
                'site': site,
                'name': name,
                'ranking': ranking,
                'schema_object': json.loads(json_str),
                'sent': False,
            }
            
            # Align early send threshold with the final desired score
            if (ranking["score"] >= self.SCORE_THRESHOLD):
                async with self._results_lock: # Protect access to early_sent_titles
                    if name not in self.early_sent_titles:
                        logger.info(f"High score item (>={self.SCORE_THRESHOLD}): {name} (score: {ranking['score']}) - sending early {self.ranking_type_str}")
                        self.early_sent_titles.add(name) # Add title before attempting to send
                        # Mark as potentially sent here, to avoid race condition with sendAnswers
                        # We'll let sendAnswers confirm if it actually sent.
                        # The main purpose of self.early_sent_titles is to avoid re-sending same title.
                        # The 'sent' flag on 'ansr' is for the final de-duplication.
                        send_this_early = True
                    else:
                        logger.info(f"High score item (>={self.SCORE_THRESHOLD}): {name} (score: {ranking['score']}) matches an already early-sent title. Skipping early send.")
                        send_this_early = False
                
                if send_this_early:
                    try:
                        await self.sendAnswers([ansr]) # sendAnswers itself has shouldSend logic
                    except (BrokenPipeError, ConnectionResetError):
                        logger.warning(f"Client disconnected while sending early answer for {name}")
                        print(f"Client disconnected while sending early answer for {name}")
                        self.handler.connection_alive_event.clear()
                        return # Don't proceed to add to rankedAnswers if connection is lost
            
            async with self._results_lock:  # Use lock when modifying shared state
                self.rankedAnswers.append(ansr)
            logger.debug(f"Item {name} added to ranked answers")
        
        except Exception as e:
            logger.error(f"Error in rankItem for {name}: {str(e)}")
            logger.debug(f"Full error trace: ", exc_info=True)
            print(f"Error in rankItem for {name}: {str(e)}")

    def shouldSend(self, result):
        should_send = False
        if (self.num_results_sent < self.NUM_RESULTS_TO_SEND - 5):
            should_send = True
        else:
            for r in self.rankedAnswers:
                if r["sent"] == True and r["ranking"]["score"] < result["ranking"]["score"]:
                    should_send = True
                    break
        
        logger.debug(f"Should send result {result['name']}? {should_send} (sent: {self.num_results_sent})")
        return should_send
    
    async def sendAnswers(self, answers, force=False):
        if not self.handler.connection_alive_event.is_set():
            logger.warning("Connection lost during ranking, skipping sending results")
            print("Connection lost during ranking, skipping sending results")
            return
        
        if (self.ranking_type == Ranking.FAST_TRACK and self.handler.abort_fast_track_event.is_set()):
            logger.info("Fast track aborted, not sending answers")
            return
              
        json_results = []
        logger.debug(f"Considering sending {len(answers)} answers (force: {force})")
        
        for result in answers:
            if self.shouldSend(result) or force:
                json_results.append({
                    "url": result["url"],
                    "name": result["name"],
                    "site": result["site"],
                    "siteUrl": result["site"],
                    "score": result["ranking"]["score"],
                    "description": result["ranking"]["description"],
                    "schema_object": result["schema_object"],
                })
                
                result["sent"] = True
            
        if (json_results):  # Only attempt to send if there are results
            # Wait for pre checks to be done using event
            await self.handler.pre_checks_done_event.wait()
            
            # if we got here, prechecks are done. check once again for fast track abort
            if (self.ranking_type == Ranking.FAST_TRACK and self.handler.abort_fast_track_event.is_set()):
                logger.info("Fast track aborted after pre-checks")
                return
            
            try:
                if (self.ranking_type == Ranking.FAST_TRACK):
                    self.handler.fastTrackWorked = True
                    logger.info("Fast track ranking successful")
                
                to_send = {"message_type": "result_batch", "results": json_results, "query_id": self.handler.query_id}
                await self.handler.send_message(to_send)
                self.num_results_sent += len(json_results)
                logger.info(f"Sent {len(json_results)} results, total sent: {self.num_results_sent}")
            except (BrokenPipeError, ConnectionResetError) as e:
                logger.error(f"Client disconnected while sending answers: {str(e)}")
                log(f"Client disconnected while sending answers: {str(e)}")
                self.handler.connection_alive_event.clear()
            except Exception as e:
                logger.error(f"Error sending answers: {str(e)}")
                log(f"Error sending answers: {str(e)}")
                self.handler.connection_alive_event.clear()
  
    async def sendMessageOnSitesBeingAsked(self, top_embeddings_dicts):
        if (self.handler.site == "all" or self.handler.site == "nlws"):
            sites_in_embeddings = {}
            for item_dict in top_embeddings_dicts:
                site = item_dict.get("site")
                if site:
                    sites_in_embeddings[site] = sites_in_embeddings.get(site, 0) + 1
            
            top_sites = sorted(sites_in_embeddings.items(), key=lambda x: x[1], reverse=True)[:3]
            top_sites_str = ", ".join([self.prettyPrintSite(x[0]) for x in top_sites])
            message = {"message_type": "asking_sites",  "message": "Asking " + top_sites_str}
            
            logger.info(f"Sending sites message: {top_sites_str}")
            
            try:
                await self.handler.send_message(message)
                self.handler.sites_in_embeddings_sent = True
            except (BrokenPipeError, ConnectionResetError):
                logger.warning("Client disconnected when sending sites message")
                print("Client disconnected when sending sites message")
                self.handler.connection_alive_event.clear()
    
    async def do(self):
        logger.info(f"Starting ranking process with {len(self.items)} items")
        tasks = []
        # Send a message on the sites being asked if this is an "all" or "nlws" query
        if ((self.handler.site == "all" or self.handler.site == "nlws") and not self.handler.sites_in_embeddings_sent):
            await self.sendMessageOnSitesBeingAsked(self.items[:10])

        for item_dict in self.items:
            if not self.handler.connection_alive_event.is_set():
                logger.warning("Connection lost, aborting ranking tasks in do() loop")
                break # Exit the loop if connection is lost
            
            url = item_dict.get("url")
            name = item_dict.get("title")  # Assuming 'title' from metadata is used as 'name'
            site_val = item_dict.get("site") # Renamed to avoid conflict with site module
            json_str = json.dumps(item_dict)
            
            # Ensure essential values are present before creating a task
            if url and name and site_val:
                tasks.append(asyncio.create_task(self.rankItem(url, json_str, name, site_val)))
            else:
                logger.warning(f"Skipping item due to missing url, title, or site: {item_dict}")

        if tasks: # Only proceed if there are tasks to run
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info("All ranking tasks completed or connection lost.")
            except Exception as e:
                logger.error(f"Error during asyncio.gather in Ranking.do: {e}", exc_info=True)
        else:
            logger.info("No ranking tasks were created or connection was lost before loop.")

        async with self._results_lock:
            # De-duplicate self.rankedAnswers based on URL, keeping the one with the highest score
            seen_urls = {}
            deduplicated_ranked_answers_by_url = [] # Renamed for clarity
            # Sort by score descending to keep the highest score for a given URL
            sorted_for_url_dedup = sorted(self.rankedAnswers, key=lambda x: x.get('ranking', {}).get('score', 0), reverse=True)

            original_ranked_count = len(self.rankedAnswers) # Store original count for logging

            for item in sorted_for_url_dedup:
                url = item.get('url')
                # Ensure URL is a non-empty string before using it as a key
                if url and isinstance(url, str) and url.strip() != "":
                    if url not in seen_urls:
                        deduplicated_ranked_answers_by_url.append(item)
                        seen_urls[url] = True
                    else:
                        logger.debug(f"Duplicate URL found and skipped during de-duplication: {url} (Score: {item.get('ranking', {}).get('score', 0)})")
                else:
                    # If URL is missing or invalid, we might still want to keep the item if it's unique in other ways,
                    # or decide to discard it. For now, let's add it if no valid URL.
                    # This case should be rare if ingestion ensures URLs.
                    deduplicated_ranked_answers_by_url.append(item) # Add item even if URL is problematic, to not lose it yet
                    logger.warning(f"Item encountered with missing or invalid URL during de-duplication: {item.get('name')}")
            
            self.rankedAnswers = deduplicated_ranked_answers_by_url # Items are now unique by URL (highest score chosen)

            # Filter for score, then de-duplicate by title (most informative), then ensure not re-sending early-sent items
            
            # 1. Filter items that meet the score threshold from the URL-deduplicated list
            relevant_items_after_url_dedup = [
                item for item in self.rankedAnswers # self.rankedAnswers is now deduplicated_ranked_answers_by_url
                if item.get('ranking', {}).get('score', 0) >= self.SCORE_THRESHOLD
            ]
            
            # 2. Group by title and select the best item (highest score, then longest description)
            items_grouped_by_title = {}
            for item in relevant_items_after_url_dedup:
                title = item.get('name')
                if title: # Ensure title exists
                    if title not in items_grouped_by_title:
                        items_grouped_by_title[title] = []
                    items_grouped_by_title[title].append(item)
                else: # Should ideally not happen if items always have names/titles
                    logger.warning(f"Item without a title encountered during title de-duplication: {item.get('url')}")
                    # Decide how to handle items without titles: for now, add them directly if they scored well
                    # as they can't be de-duplicated by title.
                    # final_candidate_list.append(item) # Or skip them

            final_candidate_list = []
            for title, grouped_items in items_grouped_by_title.items():
                if len(grouped_items) == 1:
                    final_candidate_list.append(grouped_items[0])
                else:
                    # Multiple items with same title (different URLs, all score >= SCORE_THRESHOLD)
                    # Sort by score (desc), then description length (desc)
                    best_item = sorted(
                        grouped_items, 
                        key=lambda x: (
                            x.get('ranking', {}).get('score', 0), 
                            len(x.get('ranking', {}).get('description', ''))
                        ), 
                        reverse=True
                    )[0]
                    final_candidate_list.append(best_item)
                    logger.info(f"For title '{title}', selected item (URL: {best_item.get('url')}) with score {best_item.get('ranking',{}).get('score')} and desc length {len(best_item.get('ranking',{}).get('description',''))} over {len(grouped_items)-1} other variants.")

            # Add items that had no title but were relevant (if any were collected separately)
            # For simplicity, items without title are currently just logged and not added to final_candidate_list from items_grouped_by_title path.
            # If an item had no title but was in relevant_items_after_url_dedup, it wouldn't be in items_grouped_by_title.
            # Let's ensure such items are also considered if they exist.
            items_without_titles_but_relevant = [
                item for item in relevant_items_after_url_dedup if not item.get('name')
            ]
            final_candidate_list.extend(items_without_titles_but_relevant)
            if items_without_titles_but_relevant:
                logger.info(f"Added {len(items_without_titles_but_relevant)} relevant items that had no title to the final candidate list.")


            # 3. From this refined list, send only those not already early-sent
            truly_final_items_to_send = []
            # items_ge_threshold_count = len(final_candidate_list) # This is the count after all de-duplication

            for item in final_candidate_list: # these are already filtered for score and de-duplicated by URL and Title
                title = item.get('name')
                # Check if this specific item instance was early-sent (via its own 'sent' flag)
                # AND if its title was part of *any* early send (via self.early_sent_titles)
                # We only send if this instance wasn't sent AND no item with this title was early-sent.
                if not item.get("sent", False) and (not title or title not in self.early_sent_titles):
                    truly_final_items_to_send.append(item)
                    # Optionally mark item as sent now, though it won't be processed further here.
                    # item["sent"] = True 
                else:
                    if item.get("sent", False):
                        # This specific item instance was already marked as 'sent' (likely an early send itself)
                        logger.info(f"Item '{title or item.get('url')}' (Instance was early-sent) score >= {self.SCORE_THRESHOLD}. Not adding to final batch.")
                    elif title and title in self.early_sent_titles:
                        # This specific item instance was NOT early-sent, but another item with the SAME TITLE WAS.
                        # So, we suppress this one too to avoid sending multiple versions of the same conceptual item.
                        logger.info(f"Item '{title}' (Title matches an item that was already early-sent) score >= {self.SCORE_THRESHOLD}. Not adding to final batch.")
                    # else: item has no title, and wasn't sent. Could be added if desired, but current logic filters by title.
            
            logger.info(f"Originally {original_ranked_count} items. After URL de-dup, {len(self.rankedAnswers)} items. Filtered for score >= {self.SCORE_THRESHOLD}: {len(relevant_items_after_url_dedup)} items. After title de-dup (choosing best desc/score), {len(final_candidate_list)} final candidates. Sending {len(truly_final_items_to_send)} of these that were not early-sent AND whose titles don't match any early-sent item.")

            if truly_final_items_to_send: # Only call sendAnswers if there's something new to send
                await self.sendAnswers(truly_final_items_to_send, force=True)
            else:
                logger.info(f"No new items (score >= {self.SCORE_THRESHOLD}, de-duplicated, and not early-sent) to send in the final batch.")

        logger.info("Finished Ranking.do() method")

    def prettyPrintSite(self, site):
        ans = site.replace("_", " ")
        words = ans.split()
        return ' '.join(word.capitalize() for word in words)
