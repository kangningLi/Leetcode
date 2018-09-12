/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *sum_list= new ListNode(-1);
        ListNode *p= sum_list;
        int carry=0;
        while(l1 || l2)
        {
            int n1= l1? l1->val:0;
            int n2= l2? l2->val:0;
            int sum=n1+n2+carry;
            carry= sum/10;
            p->next=new ListNode(sum % 10);
            p=p->next;
            if (l1) l1 = l1->next;
            if (l2) l2 = l2->next;
        }
        
        if (carry) p->next = new ListNode(1);
        return sum_list->next;
        
    }
};