class Solution {
public:
    string convert(string s, int numRows) {
        if (s == || s.length()<2 || numRows < 2)
            return s;
        string res;
        int position = 0;
        int size= 2*numRows -2;
        
        for(int i=0;i<numRows; i++){
            
            for(int j=0;j+i<s.length();j+=size)
            {
                res+=s[j+i];
                if (i!=0 && i!=numRows -1 && j+size-i<s.length())
                    res+=s[j+size-i];
            }
            
        }
        
        return res;
        
    }
};
