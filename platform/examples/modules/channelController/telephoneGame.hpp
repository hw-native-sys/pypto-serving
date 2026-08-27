#pragma once

#include <vector>
#include <memory>
#include <thread>

#include <hicr/core/instance.hpp>

#include <system/channels/message.hpp>
#include <system/channels/input.hpp>
#include <system/channels/output.hpp>

#define _REPLICAS_PER_PARTITION 1

__INLINE__ void telephoneGame(serving::system::channels::Input  &inputChannel,
                              serving::system::channels::Output &outputChannel,
                              const HiCR::Instance::instanceId_t instanceId,
                              const bool                         isRoot)
{
  if (isRoot)
  {
    // Root instance starts the game by sending a message to the next instance
    const std::string text = "Hello from root instance!";
    printf("[Instance %lu][TelephoneGame] Sending message: %s\n", instanceId, text.c_str());
    auto input = serving::system::channels::Message(reinterpret_cast<const uint8_t *>(text.data()), text.size(), serving::system::channels::Message::metadata_t{});
    outputChannel.pushMessageLocking(input);

    // wait for the return message
    while (inputChannel.hasMessage() == false) { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }
    auto output = inputChannel.getMessage();

    printf("[Instance %lu][TelephoneGame] Received message: %s\n", instanceId, std::string(reinterpret_cast<const char *>(output.getData()), output.getSize()).c_str());
    inputChannel.popMessage();
  }
  else
  {
    while (inputChannel.hasMessage() == false) { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }
    auto input = inputChannel.getMessage();

    printf("[Instance %lu][TelephoneGame] Received message: %s\n", instanceId, std::string(reinterpret_cast<const char *>(input.getData()), input.getSize()).c_str());

    auto text = std::string(reinterpret_cast<const char *>(input.getData()), input.getSize());
    printf("[Instance %lu][TelephoneGame] Sending message: %s\n", instanceId, text.c_str());
    auto output = serving::system::channels::Message(reinterpret_cast<const uint8_t *>(text.data()), text.size(), serving::system::channels::Message::metadata_t{});
    outputChannel.pushMessageLocking(output);
    inputChannel.popMessage();
  }
}