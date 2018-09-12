class Solution {
public:
  vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> indies;
    for (int i = 0; i < nums.size(); ++i)
      indies[nums[i]] = i;
    
    for (int i = 0; i < nums.size(); ++i) {
      int left = target - nums[i];
      if (indies.count(left) && indies[left] != i) {
          return {i, indies[left]};
      }
    }
    return {};
  }
};
