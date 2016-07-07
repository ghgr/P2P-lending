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


import pandas as pd
import operator
import sys
sys.path.append("api")
import bondora
reload(bondora)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
import datetime
from copy import copy
import os
import urllib
import time
import cPickle



numeric_keys = [#"BidsApi",
		#"BidsManual",
		"Age",
#			"TotalNumDebts",
#			"TotalMaxDebtMonths",
		"AppliedAmount",
#		"FundedAmount",
		"Interest",
#		"IssuedInterest",
		"LoanDuration",
		"income_total",
#			"DebtToIncome",
#		"AppliedAmountToIncome"
		]

date_keys = [	"LoanDate",
		"ContractEndDate",
		"FirstPaymentDate",
		"MaturityDate_Original",
		"MaturityDate_Last"
		]


boolean_keys = [ #"WasFunded",
#		"IsBusinessLoan",
		"NewCreditCustomer",
#			"1D FromFirstPayment",
#			"14D FromFirstPayment",
#			"30D FromFirstPayment",
#			"60D FromFirstPayment"
		]

categorical_keys = ["VerificationType",
		"Gender",
		"Country",
		"credit_score",
#		"CreditGroup",
		"UseOfLoan",
#		"ApplicationType",
		"education_id",
		"marital_status_id",
		"nr_of_dependants",
		"employment_status_id",
		"Employment_Duration_Current_Employer",
		"work_experience",
		"occupation_area",
		"home_ownership_type_id"
		]

used_keys = numeric_keys+categorical_keys+boolean_keys

def downloadAndSaveHistoricalData():
        filename = "LoanData_"+str(int(time.time()))+".pickle"
	print os.system("wget -O LoanData/LoanData.xls https://www.bondora.com/marketing/media/LoanData.xlsx")
	print "Done, opening XLS (takes a while)..."
	data = pd.read_excel("LoanData/LoanData.xls")
	print "Saving into",filename,"..."
	data.to_pickle("LoanData/"+filename)
	print "Done, removin excel file..."
	os.system("rm LoanData/LoanData.xls")
	print "Done!"
	return filename

def loadXY():

	try:
		filename = sorted(os.listdir('LoanData'), reverse=True)[0]
		timestamp = int(filename.split("_")[1].split(".")[0])
		if (time.time()-timestamp)>86400:
			print "Latest file,",filename,"is obsolete. Downloading new one..."
			filename = downloadAndSaveHistoricalData()
		else:
			print "Filename",filename, "is",time.time()-timestamp,"seconds old, using this one"		
	except:
		print "No files in folder, downloading..."
		filename = downloadAndSaveHistoricalData()

	data = pd.read_pickle("LoanData/"+filename)
	data.sort_values("MaturityDate_Last",inplace=True, ascending=True)
	data.dropna(axis=0, inplace=True, subset = ["AppliedAmount","MaturityDate_Last","Interest"])

	

	print "Filtering finished AND funded loans!"
#	print "Going from",data.shape[0],
	data = data[data.MaturityDate_Last<datetime.datetime.today()]
	data = data[data.WasFunded==1]
#	print "to",data.shape[0]

	for k in categorical_keys:
		data[k] = data[k].apply(hash)

	defaulted_idx= np.array(~np.isnan(data.DefaultedOnDay))
	applied_amount_idx = used_keys.index("AppliedAmount")
	interest_idx = used_keys.index("Interest")
	date_loan = data.LoanDate
	date_maturity = data.MaturityDate_Last
	X = np.array(data[used_keys])
	y = defaulted_idx
	return X,y, applied_amount_idx, interest_idx, date_loan, date_maturity

def analyzeResults(Xtest, ytest, ypred, minimum_amount, InterestTest, AppliedAmountTest):
	global probs

	action = np.logical_and( (ypred>=threshold), (AppliedAmountTest>minimum_amount))
	depth = AppliedAmountTest[action].min() * action.sum()
	if depth<money_to_invest:
		raise Exception("Not enough depth to invest "+str(money_to_invest))

	num_loans = int(money_to_invest/minimum_amount)
	initial_loans = action.sum()
	print "Money to invest: %.2f" % (money_to_invest)
	if num_loans>=action.sum():
		print "You can invest in all %d loans" % (action.sum())
	else:
		print "If you invested in all %d loans, you would put %.2f in each, which is below the minimum (%.2f). You can only invest %.2f in %d loans" % (action.sum(), 1.0*money_to_invest/action.sum(),minimum_amount, minimum_amount, num_loans)
		order_idx = np.int32(sorted(zip(ypred,np.arange(Xtest.shape[0])), key =operator.itemgetter(0), reverse=False))[:,1]

		idx_to_not_invest = order_idx[ypred[order_idx]>threshold][:initial_loans-num_loans]
		action[idx_to_not_invest] = 0
		print "So now you invest in just %d loans" % (action.sum())
 
	good = (action==False) * (ytest==False)
	bad = (action==True) * (ytest==True)

	InterestAfterDefault = InterestTest.copy()
	InterestAfterDefault[ytest] = -1
	InterestAfterDefault = InterestAfterDefault[action]
	lost_money_in_default = AppliedAmountTest[bad].sum()
	depth = AppliedAmountTest[action].min() * action.sum()

	print "\tThreshold:",threshold
	print "\t%d of the %d times I thought it was OK, I crashed hard (%.2f%%)" % (bad.sum(), float((action==True).sum()), 100*(bad).sum()/float((action==True).sum()))
	print "\tThe maximum amount of money investible maintining equal exposition to all loans (min_depth * num_loans) is %.2f" % (depth) 
	print "\t To summarize:"
	print "\t\tI invested %.2f in %d loans, a total amount of %.2f" % (int(money_to_invest*100.0/action.sum())*0.01, action.sum(), int(money_to_invest*100.0/action.sum())*0.01 *  action.sum() )
	print "\t\tOf those loans, %d defaulted" % (bad.sum())
	print "\t\tThe resultant interest is %.2f%%" % (InterestAfterDefault.mean()*100)
	print "\n\n\n"
	
def getTrainTest(myDate, date_loan, date_maturity):
	it = cross_validation.ShuffleSplit(X.shape[0],1)
	return it

def loadXtest(filename=None):
	if filename:
		data2 = pd.read_csv(filename)
	else:
		print "Loading from api..."
		data2 = b.getAuctions()	
		print "Done, loaded",data2.shape[0],"elements"
		print "Querying current bids..."
		bids = b.getBids()
		bids_ids = [] 
		for bid in bids:
			if bid['StatusCode']==1:
				bids_ids.append(bid['AuctionId'])
		print "Done, there are currently",len(bids_ids),"bids"	
		print "Dropping from the dataset"
		for bid_id in bids_ids:
			data2 = data2[data2.AuctionId!=bid_id]
		print "Done, there are",data2.shape[0],"elements"
	used_keys2 = copy(used_keys)
	categorical_keys2 = copy(categorical_keys)
	transformations = {}
	transformations['income_total'] = 'IncomeTotal'
	transformations['credit_score'] = 'CreditScore'
	transformations['education_id'] = 'Education'
	transformations['marital_status_id'] = 'MaritalStatus'
	transformations['nr_of_dependants'] = 'NrOfDependants'
	transformations['employment_status_id'] = 'EmploymentStatus'
        transformations['Employment_Duration_Current_Employer'] = 'EmploymentDurationCurrentEmployer'
        transformations['work_experience'] = 'WorkExperience'
        transformations['occupation_area'] = 'OccupationArea'
        transformations['home_ownership_type_id'] =  'HomeOwnershipType'
	data2['Interest']*=0.01
	
	for k,v in transformations.iteritems():
		try:
			used_keys2[used_keys2.index(k)] = v
			categorical_keys2[categorical_keys2.index(k)] = v
		except:
			pass
        for k in categorical_keys2:
                data2[k] = data2[k].apply(hash)

	ids = data2.AuctionId
	return ids, np.array(data2[used_keys2]), data2.ProbabilityOfDefault

def getListOfLoansToInvest(filename=None, verbose = True):
	global ypred
	ids, Xtest, pf = loadXtest(filename)
	AppliedAmountTest = Xtest[:,applied_amount_idx]			
	InterestTest = Xtest[:,interest_idx]
	est= ensemble.RandomForestClassifier(n_estimators = 100, n_jobs = -1)
	est.fit(X,y)
	probs = est.predict_proba(Xtest)[:,0]
#	probs = pf
	ypred = (np.min(zip(probs, 0.999*np.ones_like(probs)), axis=1)*(1.0+InterestTest)-1.0)/InterestTest
	action = np.logical_and( (ypred>=threshold), (AppliedAmountTest>minimum_amount))
	depth = AppliedAmountTest[action].min() * action.sum()
	if depth<money_to_invest:
		raise Exception("Not enough depth to invest "+str(money_to_invest))

	num_loans = int(money_to_invest/minimum_amount)
	initial_loans = action.sum()
	if verbose:
		print "Money to invest: %.2f" % (money_to_invest)
	if num_loans>=action.sum():
		if verbose:
			print "You can invest in all %d loans" % (action.sum())
	else:
		if verbose:
			print "If you invested in all %d loans, you would put %.2f in each, which is below the minimum (%.2f). You can only invest %.2f in %d loans" % (action.sum(), 1.0*money_to_invest/action.sum(),minimum_amount, minimum_amount, num_loans)
		order_idx = np.int32(sorted(zip(ypred,np.arange(ids.shape[0])), key =operator.itemgetter(0), reverse=False))[:,1]

		idx_to_not_invest = order_idx[ypred[order_idx]>threshold][:initial_loans-num_loans]
		action[idx_to_not_invest] = 0
		if verbose:
			print "So now you invest in just %d loans" % (action.sum())
 
	depth = AppliedAmountTest[action].min() * action.sum()
	qty = int(money_to_invest*100.0/action.sum())*0.01
	qty = int(qty/5)*5.0
	if verbose:
		
		print "\tThreshold:",threshold
		print "\tOut of the %d loans, I find %d interesting" % (action.shape[0], action.sum())
		print "\tThe maximum amount of money investible maintining equal exposition to all loans (min_depth * num_loans) is %.2f" % (depth) 
		print "\t To summarize:"
		print "\t\tI invested %.2f in %d loans, a total amount of %.2f" % ( qty, action.sum(), action.sum() *  qty)
		print "\t\tIf NO loan defaults, I expect a performance of %.2f%%" % (InterestTest[action].mean()*100)
		InterestAfterDefault = copy(InterestTest)[action]
		InterestAfterDefault[InterestAfterDefault.argmax()] = -1
		print "\t\tIf the loan with largest interest (%.2f%%) defaults, I expect a performance of %.2f%%" % (InterestTest[action].max()*100, InterestAfterDefault.mean()*100)
		InterestAfterDefault[InterestAfterDefault.argmax()] = -1
		print "\t\tIf the TWO loans with largest interests default, I expect a performance of %.2f%%" % (InterestAfterDefault.mean()*100 )

	return list(ids[action]), qty
	

if __name__=="__main__":

	TOKEN = "<YOUR_BONDORA_API_TOKEN_HERE>"
	useRealData = 1
	threshold = 0.95
	minimum_amount = 5.0


	if useRealData:
		b = bondora.Bondora(token = TOKEN)	
		money_to_invest = b.getAvailableFundsToBid() 

		assert money_to_invest>=minimum_amount, "You can't buy a single loan with %.2f" % (money_to_invest)

	else:
		money_to_invest = 100
	X,y, applied_amount_idx, interest_idx, date_loan, date_maturity= loadXY()

	if useRealData:
		print "Using REAL data"
#		l,qty = getListOfLoansToInvest(filename="openLoans.csv", verbose=True)
		l,qty = getListOfLoansToInvest(filename=None, verbose=True)
		raw_input("Press any key to invest")
		b.bidLoans(l,qty)

	else:
		print "Using MOCK data"
		myDate = datetime.datetime(2016,01,01)
		it = getTrainTest(myDate, date_loan, date_maturity) 
		for train, test in it:
	#		train = range(len(X)*7/8)
	#		test = range(len(X)*7/8,len(X))

			Xtrain = X[train]
			Xtest = X[test]
			ytrain = y[train]
			ytest = y[test] 

			AppliedAmountTest = Xtest[:,applied_amount_idx]			
			InterestTest = Xtest[:,interest_idx]*0.01

			print "General info about the test set:"
			print "\t%d loans" % (ytest.shape[0])
			print "\t%d defaults" % (ytest.sum())
			print "\t%.2f%% interest in non-defaulted loans" % (100*InterestTest[~ytest].mean())
			print "\t%.2f%% interest in YES-defaulted loans (ofc this money was NEVER paid back)" % (100*InterestTest[ytest].mean())
			print "\n"


			print "\nBenchmark 1) Always invest"
			ypred = np.ones_like(ytest)*1.0
			analyzeResults(Xtest, ytest, ypred, minimum_amount, InterestTest, AppliedAmountTest)
			print "\nBenchmark 2) Always invest in hindsight in successful loans"
			ypred = np.ones_like(ytest)*1.0
			ypred[ytest] = 0.0
			analyzeResults(Xtest, ytest, ypred, minimum_amount, InterestTest, AppliedAmountTest)

			print "\nActual 1) Random Forests with 100 trees"
			est= ensemble.RandomForestClassifier(n_estimators = 100, n_jobs = -1)
			est.fit(Xtrain,ytrain)
			probs = est.predict_proba(Xtest)[:,0]
			ypred = (np.min(zip(probs, 0.999*np.ones_like(probs)), axis=1)*(1.0+InterestTest)-1.0)/InterestTest
			analyzeResults(Xtest, ytest, ypred, minimum_amount, InterestTest, AppliedAmountTest)

			if False:
				print "\nActual 2) XGBoost" 
				import xgboost
				est= xgboost.XGBClassifier(n_estimators=100) 
				est.fit(Xtrain,ytrain)
				ypred = (est.predict_proba(Xtest)[:,0]*(1.0+InterestTest)-1.0)/InterestTest	
				analyzeResults(Xtest, ytest, ypred, minimum_amount, InterestTest, AppliedAmountTest)

