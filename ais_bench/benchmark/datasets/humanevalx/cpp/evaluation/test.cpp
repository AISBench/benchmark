/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
separate those group into separate strings and return the vector of those.
Separate groups are balanced (each open brace is properly closed) and not nested within each other
Ignore any spaces in the input string.
>>> separate_paren_groups("( ) (( )) (( )( ))")
{"()", "(())", "(()())"}
*/
#include <cstdio>
#include <vector>
#include <string>

namespace {

    std::vector<std::string> separate_paren_groups(std::string paren_string)
    {
        std::vector<std::string> all_parens;
        std::string current_paren;
        int level = 0;
        char chr;
        int i;

        for (i = 0; i < paren_string.length(); i++) {
            chr = paren_string[i];
            if (chr == '(') {
                level += 1;
                current_paren += chr;
            }
            if (chr == ')') {
                level -= 1;
                current_paren += chr;
                if (level == 0) {
                    all_parens.push_back(current_paren);
                    current_paren = "";
                }
            }
        }
        return all_parens;
    }

#undef NDEBUG
#include <cassert>

    static bool g_issame(std::vector<std::string> a, std::vector<std::string> b)
    {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
} // namespace

int main()
{
    assert(g_issame(separate_paren_groups("(()()) ((())) () ((())()())"), {"(()())", "((()))", "()", "((())()())"}));
    assert(g_issame(separate_paren_groups("() (()) ((())) (((())))"), {"()", "(())", "((()))", "(((())))"}));
    assert(g_issame(separate_paren_groups("(()(())((())))"), {"(()(())((())))"}));
    assert(g_issame(separate_paren_groups("( ) (( )) (( )( ))"), {"()", "(())", "(()())"}));
}