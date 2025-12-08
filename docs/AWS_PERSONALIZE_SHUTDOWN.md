# AWS Personalize Shutdown Documentation

## 📅 Date: December 8, 2025

## 🎯 Purpose
Shut down AWS Personalize services to reduce costs while maintaining recommendation functionality through cached data.

---

## 💰 Cost Analysis Before Shutdown

| Resource | Status | Monthly Cost |
|----------|--------|--------------|
| **Campaign** (mastergroup-campaign) | ACTIVE | ~$144/month |
| **Daily Batch Jobs** | Running at 2AM daily | ~$25/month |
| **Data Storage** | ACTIVE | ~$2/month |
| **Total Monthly Cost** | | **~$170/month** |

### Cost Breakdown from AWS Cost Explorer:
- **November 2025**: $24.75 (BatchInference: $24.75)
- **Projected Annual Cost**: ~$2,040/year

---

## ✅ Actions Taken

### 1. Campaign Deletion
**Command Executed:**
```bash
aws personalize delete-campaign \
  --campaign-arn arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign \
  --region us-east-1
```

**Result:** Campaign deletion initiated successfully
**Savings:** ~$144/month

### 2. Batch Job Disabled
**Original Crontab:**
```bash
0 2 * * * /opt/mastergroup-api/run_daily_batch.sh
0 */6 * * * /opt/mastergroup-api/run_sync.sh
```

**Updated Crontab:**
```bash
# AWS Personalize batch job disabled on Mon Dec 8 23:05:00 PKT 2025 to save costs
# 0 2 * * * /opt/mastergroup-api/run_daily_batch.sh
0 */6 * * * /opt/mastergroup-api/run_sync.sh
```

**Result:** Daily batch inference job disabled
**Savings:** ~$25/month

---

## 📦 Cached Data Preserved

The following recommendation data is cached in the database and will continue to work:

| Table | Records | Last Updated |
|-------|---------|--------------|
| `offline_user_recommendations` | 180,483 users | Dec 8, 2025 03:00 AM |
| `offline_similar_items` | Available | Dec 8, 2025 |
| `recommendation_cache` | Available | Dec 8, 2025 |

### Database Location:
- **Host:** `ls-49a54a36b814758103dcc97a4c41b7f8bd563888.cijig8im8oxl.us-east-1.rds.amazonaws.com`
- **Database:** `mastergroup_recommendations`
- **Port:** 5432

---

## 🔧 AWS Resources Status After Shutdown

### Deleted:
- ❌ Campaign: `mastergroup-campaign`

### Still Active (for potential future use):
- ⚠️ Solutions: `mastergroup-item-affinity`, `mastergroup-similar-items`, `mastergroup-user-personalization`
- ⚠️ Dataset Group: `mastergroup-recommendations`
- ⚠️ Datasets: Interactions, Items, Users

### To Completely Remove AWS Personalize (Optional):
```bash
# Delete all solutions (after campaign is fully deleted)
aws personalize delete-solution --solution-arn arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-item-affinity
aws personalize delete-solution --solution-arn arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-similar-items
aws personalize delete-solution --solution-arn arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-user-personalization

# Delete dataset group (after solutions are deleted)
aws personalize delete-dataset-group --dataset-group-arn arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations
```

---

## 🚀 Impact on Application

### What Still Works:
- ✅ Dashboard displays cached recommendations
- ✅ API returns cached user recommendations
- ✅ Similar items functionality (from cache)
- ✅ All analytics and reporting
- ✅ Data sync continues (every 6 hours)

### What Stops Working:
- ❌ Real-time AWS Personalize API calls
- ❌ Daily recommendation refresh
- ❌ New user recommendations (for users not in cache)

### Mitigation:
The custom ML model is available as a replacement:
- Location: `/opt/mastergroup-api/src/algorithms/`
- Capabilities: Collaborative filtering, content-based, hybrid recommendations
- Cost: $0 (runs on existing infrastructure)

---

## 📊 Estimated Savings

| Period | Before | After | Savings |
|--------|--------|-------|---------|
| Monthly | $170 | ~$2 (storage only) | **$168/month** |
| Annual | $2,040 | ~$24 | **$2,016/year** |

---

## 🔄 How to Re-enable (If Needed)

### 1. Re-enable Batch Jobs:
```bash
# SSH to server
ssh -i LightsailDefaultKey-us-east-1.pem ubuntu@44.201.11.243

# Edit crontab
crontab -e

# Uncomment the batch job line:
0 2 * * * /opt/mastergroup-api/run_daily_batch.sh
```

### 2. Recreate Campaign:
```bash
aws personalize create-campaign \
  --name mastergroup-campaign \
  --solution-version-arn <solution-version-arn> \
  --min-provisioned-tps 1 \
  --region us-east-1
```

---

## 📝 Notes

1. **Campaign deletion takes time** - AWS Personalize campaigns can take 15-30 minutes to fully delete
2. **Solutions preserved** - The trained models are still available if you need to recreate the campaign
3. **Data preserved** - All training data and cached recommendations are preserved
4. **Custom ML ready** - The custom ML model can serve as a complete replacement

---

## 👤 Performed By
- **Date:** December 8, 2025
- **Time:** 23:05 PKT
- **Server:** 44.201.11.243 (Lightsail)

---

## 📎 Related Documents
- [Custom ML Playbook](../custom_ml/CUSTOM_ML_PLAYBOOK.md)
- [AWS Personalize Playbook](../aws_personalize/PLAYBOOK.md)
- [Client Demo Script](../CLIENT_DEMO_SCRIPT.md)
