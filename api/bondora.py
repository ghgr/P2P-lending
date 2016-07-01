"""
The MIT License (MIT)

Copyright (c) 2016 Eduardo Pena Vina

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import requests
import json
import pandas as pd

class Bondora:

	def __init__(self, token):
		self.base_url = "https://api.bondora.com/"
		self.token = token
                self.json_encoder = json.JSONEncoder()
                self.json_decoder = json.JSONDecoder()
		self.headers = {"Authorization" : "Bearer "+ self.token, "Content-Type" : "text/json"}
		self.BID_STATUS = { 	0 : "PENDING",
					1 : "OPEN",
					2 : "SUCCESSFUL",
					3 : "FAILED",
					4 : "CANCELLED",
					5 : "ACCEPTED"
				}


	def _get(self, url):
		response = requests.get(url, headers = self.headers)
		return self.json_decoder.decode(response.content)

        def _post(self, url, data):
                response = requests.post(url, data = self.json_encoder.encode(data), headers = self.headers)
		return self.json_decoder.decode(response.content)

	def _standardGetter(self, path):
		url = self.base_url+path
		response = self._get(url)
		if response['Errors']:
			raise Exception("Error in request: "+str(response))
		return response['Payload']

	def _standardPoster(self, path, data):
		url = self.base_url+path
		response = self._post(url, data)
		if response['Errors']:
			raise Exception("Error in request: "+str(response))
		return response['Payload']

	def getBalance(self):
		return self._standardGetter("api/v1/account/balance")['Balance']
	def getBids(self):
		bids = self._standardGetter("api/v1/bids")
		for i in range(len(bids)):
			bids[i]['Status'] = self.BID_STATUS[bids[i]['StatusCode']]
		return bids

	def getInvestments(self):
		return self._standardGetter("api/v1/account/investments")
	def getAuctions(self, csv_file=None):
		auctions = pd.DataFrame(self._standardGetter("api/v1/auctions"))
		if csv_file:
			auctions.to_csv(csv_file, encoding='utf-8') 
		return auctions
	def getSecondaryMarket(self, csv_file=None):
		auctions = pd.DataFrame(self._standardGetter("api/v1/secondarymarket"))
		if csv_file:
			auctions.to_csv(csv_file, encoding='utf-8') 
		return auctions
	def bidLoans(self,loans_ids, qty):
		assert type(loans_ids) == list, "loans_ids must be a list"
		bids = [{'AuctionId':loan_id, 'Amount':qty, 'MinAmount':5.0} for loan_id in loans_ids]
#		raise Exception("I'm not bidding more!")
		response = self._standardPoster("api/v1/bid", {'Bids' : bids})
		return response
	def getAvailableFundsToBid(self):
		bids = self.getBids()
		unavailable_funds = 0
		for bid in bids:
			if bid['Status'] in ["OPEN","PENDING","ACCEPTED"]:
				unavailable_funds+=bid['RequestedBidAmount']
		return self.getBalance()-unavailable_funds


