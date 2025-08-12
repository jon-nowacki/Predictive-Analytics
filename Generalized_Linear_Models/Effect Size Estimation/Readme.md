Use a generalized linear model to solve this problem.   I have this data:  

* SKU_NBR: Sku number
* SKU_COST: sku cost per unit sold
* STORE_NBR: store number
* BOH: balance on hand or inventory units at the end of the week.
* YEARLY_SALES: total units sold per year
* ML_FORECAST: Machine learning forecast that week
* MULTIPLIER: a multiplier that we use to boost inventory during promo weeks. It's normally ML_FORECAST* MULTIPLIER is inventory allocation for the promotional or sales week.
* PROMO_WEEK: is this sku on promo this week. If yes, then it has a multiplier. if no then then the multiplier for ML_FORECAST is 1.

 
I'd like to know how well the multiplier (MULTIPLIER) predicts the increase in total inventory costs (SKU_COST * BOH) the week during a sale. Keep in mind there are many SKU_NBR and they all cost different amounts and they all have different sales velocity YEARLY_SALES. And WEEK's not on promotion PROMO_WEEK = 'REG" do not have a multiplier.
 
How to create a GLM that shows what fraction of BOH increase during a particular week is due to the multiplier?
