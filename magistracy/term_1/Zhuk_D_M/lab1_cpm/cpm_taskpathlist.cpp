#include "cpm.h"

#include <iostream>

void TaskPathList::Create(const TaskList &tasklist) {
    const std::vector<std::shared_ptr<Task>> &initialTasks = tasklist.GetInitialTasks();

    for (size_t iTask = 0; iTask < initialTasks.size(); ++iTask) {
        TaskPath initialPath;
        initialPath.seq.emplace_back(initialTasks[iTask]);
        pathes.push_back(std::move(initialPath));
    }

    for (size_t iPath = 0; iPath < pathes.size(); ++iPath) {
        size_t initPathSize = pathes[iPath].seq.size() - 1;

        for (size_t iTask = initPathSize; iTask < pathes[iPath].seq.size(); ++iTask) {
            const Task &task = *pathes[iPath].seq[iTask];
            pathes[iPath].duration += task.duration;

            if (task.children.empty())
                break;

            for (size_t iChild = 1; iChild < task.children.size(); ++iChild) {
                pathes.emplace_back(pathes[iPath]);
                pathes.back().seq.emplace_back(task.children[iChild]);
            }

            pathes[iPath].seq.emplace_back(task.children.front());
        }

        if (pathes[iPath].duration > maxPathDuration) {
            maxPathDuration = pathes[iPath].duration;
            iCriticalPath   = iPath;
        }
    }
}

void TaskPathList::FindTaskReserves() {

    for (size_t iLayer = 0; iLayer < pathes[iCriticalPath].seq.size(); ++iLayer) {
        size_t critPathLength = pathes[iCriticalPath].seq.size();
        const Task &critTask  = *pathes[iCriticalPath].seq[iLayer];

        for (size_t iPath = 0; iPath < pathes.size(); ++iPath) {
            if (pathes[iPath].seq.size() < (iLayer + 1))
                continue;

            if (iPath == iCriticalPath)
                continue;

            Task &t = *pathes[iPath].seq[iLayer];

            if (t.taskId == critTask.taskId)
                continue;

            if (t.children.empty()) {
                size_t taskDelta = critPathLength - iLayer;
                int critDuration = 0;

                for (size_t iTask = 0; iTask < taskDelta; ++iTask)
                    critDuration += pathes[iCriticalPath].seq[critPathLength - iTask - 1]->duration;

                t.reserve = critDuration - t.duration;
            } else {
                t.reserve =  critTask.duration - t.duration;
            }            
        }
    }
}

void TaskPathList::Dump() const {
    std::cout << "--------------- PathList ---------------" << std::endl;

    for (size_t iPath = 0; iPath < pathes.size(); ++iPath) {
        std::cout << "Path " << iPath << ": ";
        const TaskPath &path = pathes[iPath];

        for (size_t iTask = 0; iTask < path.seq.size(); ++iTask) {
            const Task &task = *path.seq[iTask];
            std::cout << task.taskId
                << ((iTask < path.seq.size() - 1) ? " --> " : "");
        }

        std::cout << ";\tduration: " << path.duration
            << ((iPath == iCriticalPath) ? " <<< CRITICAL" : "") << std::endl;
    }
}
