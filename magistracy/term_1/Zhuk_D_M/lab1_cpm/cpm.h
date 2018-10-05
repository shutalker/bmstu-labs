#ifndef LAB1_CPM_CPM_H
#define LAB1_CPM_CPM_H

#include <string>
#include <vector>
#include <map>
#include <memory>

struct Task {
    bool isInitial;
    int  duration;
    int  reserve;
    std::string taskId;
    std::vector<std::shared_ptr<Task>> children;

    Task(const std::string &id): taskId(id), isInitial(false), duration(0), reserve(0) {}
    Task(): isInitial(false), duration(0), reserve(0) {}

    bool IsInitialized() const { return isInitialized; }
    void SetInitialized()      { isInitialized = true; }

private:
    bool isInitialized = false;
};

class TaskList {
public:
    void LoadFromJsonFile(const std::string &filename) noexcept(false);
    bool IsValid() const;
    void Dump() const;
    void DumpReverseTaskLinks() const;
    const std::vector<std::shared_ptr<Task>> & GetInitialTasks() const;

private:
    std::map<std::string, std::shared_ptr<Task>> tasks;
    std::vector<std::shared_ptr<Task>>           initialTasks;
    std::map<std::string, std::vector<std::shared_ptr<Task>>> reverseTaskLinks;
};

class TaskPathList {
public:
    TaskPathList(): iCriticalPath(-1), maxPathDuration(0) {}
    void Create(const TaskList &tasklist);
    void FindTaskReserves();
    void Dump() const;

private:
    struct TaskPath {
        std::vector<std::shared_ptr<Task>> seq;
        int duration;

        TaskPath(): duration(0) {}
    };

    std::vector<TaskPath> pathes;
    int iCriticalPath;
    int maxPathDuration;
};

#endif // #ifndef LAB1_CPM_CPM_H