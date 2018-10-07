#ifndef LAB1_CPM_CPM_H
#define LAB1_CPM_CPM_H

#include <string>
#include <vector>
#include <map>
#include <deque>
#include <memory>
#include <bitset>


struct Task {
    bool isInitial = false;
    int  duration  = 0;
    int  reserve   = 0;
    int  earliestStart = 0;
    int  latestFinish  = 0;
    std::string taskId;
    std::vector<std::shared_ptr<Task>> children;
    std::vector<std::shared_ptr<Task>> parents;

    Task(const std::string &id): taskId(id) {}
    Task() {}

    bool IsParsed() const { return isParsed; }
    void SetParsed()      { isParsed = true; } 

private:
    bool isParsed = false;
};

class TaskList {
public:
    TaskList(): criticalPathDuration(0) { Init(); }
    void LoadFromJsonFile(const std::string &filename) noexcept(false);
    bool IsValid() const;
    void Dump(const std::string &suffix = "") const;
    void FindCriticalPath();
    void DumpCriticalPath();

private:
    int criticalPathDuration;
    std::map<std::string, std::shared_ptr<Task>> tasks;
    std::deque<std::shared_ptr<Task>>            criticalPath;
    std::shared_ptr<Task>                        initialTask;
    std::shared_ptr<Task>                        finalTask;   

    const std::string STR_INIT_TASK = ">>";
    const std::string STR_FIN_TASK  = "<<";

    void Init();
    void ForwardTaskLookup();
    void BackwardTaskLookup();
};

#endif // #ifndef LAB1_CPM_CPM_H